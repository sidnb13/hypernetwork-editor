from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
)

from .config import GPT2EditorConfig


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx, init_bias: bool = False):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        # Modification from original GPT2 implementation
        if init_bias:
            nn.init.normal_(self.bias, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # print device of all components
        # print(self.bias.device)
        # print(x.view(-1, x.size(-1)).device)
        # print(self.weight.device)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class EditorUnembedCrossAttention(GPT2Attention):
    is_cross_attention = True
    _flash_attn_enabled = False

    def __init__(self, config: GPT2EditorConfig, layer_idx=None, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.n_embd
        self.num_heads = config.num_editing_heads
        self.restrict_edit_to_layers = config.restrict_edit_to_layers
        self.restrict_edit_to_positions = config.restrict_edit_to_positions

        assert (
            self.num_heads % config.edit_channel_multiply_factor == 0
        ), f"num_editing_heads ({config.num_editing_heads}) must be divisible by edit_channel_width ({config.edit_channel_multiply_factor})"

        self.heads_per_multiply = self.num_heads // config.edit_channel_multiply_factor

        self.head_dim = (
            self.embed_dim * config.edit_channel_multiply_factor // self.num_heads
        )

        # # split additional factor of channel width
        # # Changing this back to embed_dim, now that we're accumulating multiplies!
        self.split_size = self.embed_dim  # * self.config.edit_channel_multiply_factor

        if (
            self.head_dim * self.num_heads
            != self.embed_dim * self.config.edit_channel_multiply_factor
        ):
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = False  # TODO: look into this
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = nn.ModuleList(
            [
                Conv1D(
                    2 * self.embed_dim,
                    self.embed_dim,
                    init_bias=config.init_attn_proj_bias,
                )
                for _ in range(config.edit_channel_multiply_factor)
            ]
        )
        self.q_attn = nn.ModuleList(
            [
                Conv1D(
                    self.embed_dim,
                    self.embed_dim,
                    init_bias=config.init_attn_proj_bias,
                )
                for _ in range(config.edit_channel_multiply_factor)
            ]
        )
        self.c_proj = nn.ModuleList(
            [
                Conv1D(
                    self.embed_dim,
                    self.embed_dim,
                    init_bias=config.init_attn_proj_bias,
                )
                for _ in range(config.edit_channel_multiply_factor)
            ]
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    def _split_heads(self, tensor: torch.Tensor):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (self.heads_per_multiply, self.head_dim)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # NOTE: Squeeze the key dimension since we want to attend over query tokens
        attn_weights = torch.matmul(query, key.transpose(-1, -2)).squeeze(-1)

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full(
                [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device
            )
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # unsqueeze to retain key length of 1
        attn_output = torch.matmul(attn_weights.unsqueeze(-1), value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(
        self, query, key, value, attention_mask=None, head_mask=None
    ):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(
            bsz * num_heads,
            q_seq_len,
            1,
            dtype=torch.float32,
            device=query.device,
        )

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = (
                query.reshape(-1, q_seq_len, dk),
                key.transpose(-1, -2).reshape(-1, dk, 1),
            )
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor
            ).squeezed(-1)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
                attn_weights.device
            )
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
        else:
            raise ValueError("This class is only meant to be used as cross attention")
            # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        for i in range(self.config.edit_channel_multiply_factor):
            query = self.q_attn[i](encoder_hidden_states)
            # We only take the last position hidden state from the editor
            key, value = self.c_attn[i](hidden_states[:, -1, :]).split(
                self.split_size, dim=-1
            )
            # print(encoder_hidden_states.shape) #torch.Size([8, 104, 768]) #I believe this is the result of torch.stacking a [8, 8, 13, 768] tensor along d=2
            # To construct the mask, we can write the mask in matrix form and then stack along d = 2

            if self.restrict_edit_to_layers != []:
                # initialize the mask
                mask = torch.zeros(
                    encoder_hidden_states.shape[0],
                    encoder_hidden_states.shape[1] // 13,
                    13,
                )  # hard-coding the target model's layer count

                for layer in self.restrict_edit_to_layers:
                    mask[:, :, layer] = 1

                for position in self.restrict_edit_to_positions:
                    mask[:, position, :] = 1

                # Havne't checked, but this next line should be effectively stacking the mask along d=2
                mask = mask.reshape(encoder_hidden_states.shape[0], -1)

                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask * mask
                else:
                    encoder_attention_mask = mask

            attention_mask = encoder_attention_mask

            split_query = self._split_heads(query)
            bsz, query_len, num_heads, head_dim = split_query.size()

            # (bsz, seq_len, num_heads, head_dim) -> (bsz, num_heads, seq_len, head_dim)
            query = split_query.permute(0, 2, 1, 3)
            key = self._split_heads(key).unsqueeze(-2)
            value = self._split_heads(value).unsqueeze(-2)

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            if self.reorder_and_upcast_attn:
                attn_output, attn_weights = self._upcast_and_reordered_attn(
                    query, key, value, attention_mask, head_mask
                )
            else:
                attn_output, attn_weights = self._attn(
                    query, key, value, attention_mask, head_mask
                )

            # unmerge the head and batch dimension
            attn_output = attn_output.reshape(bsz, num_heads, -1, head_dim)
            attn_output = self._merge_heads(
                attn_output, self.heads_per_multiply, self.head_dim
            )
            attn_output = self.c_proj[i](attn_output)
            attn_output = self.resid_dropout(attn_output)

            if i == 0:
                outputs = (attn_output, present)
                if output_attentions:
                    outputs += (attn_weights,)
            else:
                if use_cache is True:
                    raise ValueError(
                        "Error, key-value caching for this is not implemented. Should we even be doing this? -Mike"
                    )
                if output_attentions:
                    # Check stacking dimensions!
                    # Find which dimension of attn_weights is equal to the number of heads per multiply
                    # Then stack along that dimension
                    # Don't use number of heads equal to 786 until this is cleared up!
                    stacking_dim = attn_weights.shape.index(self.heads_per_multiply)
                    attn_output_old, _, attn_weights_old = outputs
                    attn_output += attn_output_old

                    attn_weights = torch.stack(
                        (attn_weights_old, attn_weights), dim=stacking_dim
                    )
                    outputs = (attn_output, present, attn_weights)
                else:
                    attn_output_old, _ = outputs
                    attn_output += attn_output_old
                    outputs = (attn_output, present)

        return outputs  # a, present, (attentions)


class OldEditorAttention(nn.Module):
    def __init__(self, config: GPT2EditorConfig):
        super().__init__()

        # Controls whether the head will do a global softmax in all positions & layers
        # If True, the attn is global and will sum to 1
        # If False, the attn is a logistic fxn independently for every layer & token
        # I suspect we will also want to penalize the intervention norm
        self.num_editing_heads = (
            config.num_editing_heads
        )  # should default to 1, but we're going to test adding more
        self.edit_channel_width = config.n_embd * config.edit_channel_multiply_factor
        if self.edit_channel_width % self.num_editing_heads != 0:
            print("Error: config hidden size is not divisible by num_editing_heads")
        self.head_dim = self.edit_channel_width // self.num_editing_heads
        self.embed_dim = config.n_embd

        max_positions = (
            config.n_positions
        )  # does this do anything? can try killing this later
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        # We compute Q and K as a single nn.linear; but will later break apart into subcomponents

        ## Before modification to a variable channel-width
        # self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        # self.k_attn = nn.Linear(self.embed_dim, self.embed_dim)
        # self.v_attn = nn.Linear(self.embed_dim, self.embed_dim)
        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.q_attn = nn.Linear(self.embed_dim, self.edit_channel_width)
        self.k_attn = nn.Linear(self.embed_dim, self.edit_channel_width)
        self.v_attn = nn.Linear(self.embed_dim, self.edit_channel_width)
        self.out_proj = nn.Linear(self.edit_channel_width, self.embed_dim)

    def _split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim)."""
        new_shape = x.size()[:-1] + (self.num_editing_heads, self.head_dim)
        return x.view(*new_shape)

    def _new_reverse_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Assume that we are doing softmax attention
        # Project and split the query, key, value tensors
        split_query = self._split_heads(query)
        split_key = self._split_heads(key)
        split_value = self._split_heads(value)

        # Double-application (is this actually good/better for some reason?)
        # self._split_heads(self.q_attn(query))
        # self._split_heads(self.k_attn(key))
        # self._split_heads(self.v_attn(value))

        if split_query.dim() != 4:
            print("Error: Expected query to be 4D tensor, but got something else!")
        if split_key.dim() != 3:
            print("Error: Expected key to be 3D tensor, but got something else!")
        if split_value.dim() != 3:
            print("Error: Expected value to be 3D tensor, but got something else!")

        # Query should be shaped as (batch_index, sequence_index, head_index, head_dim)
        # Key and value should be shaped as (batch_index, head_index, head_dim)

        KQ_weights = torch.matmul(
            split_query.permute(0, 2, 1, 3), split_key.unsqueeze(-1)
        ).squeeze(-1)

        # Then we take the softmax within the positional divisions
        softmaxed_weights = nn.functional.softmax(KQ_weights, dim=-1)

        # Adjusting value selection for head dimension
        attn_output = torch.matmul(
            softmaxed_weights.unsqueeze(-1), split_value.unsqueeze(-2)
        )

        # combine heads: change 50, 8, 104, 96 to 50, 104, 768
        # first, permute
        attn_output = attn_output.permute(0, 2, 1, 3)

        # combin heads x head_dims
        attn_output = attn_output.reshape(
            -1, attn_output.size(1), attn_output.size(2) * attn_output.size(3)
        )
        # now project back
        projected_output = self.out_proj(attn_output)

        return projected_output, softmaxed_weights

    def _reverse_attn(self, query, key, value, attention_mask=None, head_mask=None):
        if key.dim() == 4:
            K_reduced = key[
                :, :, -1, :
            ]  # R# Check: that the second dimension of K is only a single element when we have batching
            KQ_weights = torch.bmm(K_reduced, query.transpose(1, 2))
            logistic_weights = torch.atan(KQ_weights)
            attn_output = torch.bmm(
                logistic_weights.transpose(1, 2),
                value[
                    :, :, -1, :
                ],  # we take the editor output only over the final token position
            )

        if key.dim() == 3:
            QK_weights = torch.matmul(query, key.transpose(-1, -2))
            logistic_weights = torch.atan(QK_weights)
            attn_output = torch.matmul(logistic_weights, value)

        return attn_output, logistic_weights

    def forward(
        self,
        editor_hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # Here, the query is the target hidden encoder, the key is the editor, and the value is the editor
        query = self.q_attn(encoder_hidden_states)
        if editor_hidden_states.dim() == 3:
            key = self.k_attn(
                # I don't quite understand why sometimes editor_hidden_states is 4 dimensional, sometimes 3
                # seems like it's sometimes 20, 1, 4, 768 and sometimes 20, 4, 768. what gives?
                editor_hidden_states[:, -1, :]
            )  # Pull only the final token position
            value = self.v_attn(
                # [:, 0, :1, :]
                editor_hidden_states[:, -1, :]
            )  # Pull only the final token position

        if editor_hidden_states.dim() == 4:
            key = self.k_attn(
                editor_hidden_states[:, 0, -1, :]
            )  # Pull only the final token position
            value = self.v_attn(
                # [:, 0, :1, :]
                editor_hidden_states[:, 0, -1, :]
            )  # Pull only the final token position

        attn_output, attn_weights = self._new_reverse_attn(query, key, value)

        if output_attentions:
            return (attn_output, None, attn_weights)
        else:
            return (attn_output, None)
