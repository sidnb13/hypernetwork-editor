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


class GPT2EditorUnembedCrossAttention(GPT2Attention):
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
            attn_weights = attn_weights + (
                -1e9 * (1 - attention_mask.unsqueeze(1))
            )  # Mike recently implemented this. Does this look right, Sid?

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

            if (
                self.config.restrict_edit_to_layers != []
                or self.config.restrict_edit_to_positions != []
            ):
                # initialize the mask
                mask = torch.ones(
                    encoder_hidden_states.shape[0],
                    encoder_hidden_states.shape[1] // self.config.n_layer,
                    self.config.n_layer,
                )  # hard-coding the target model's layer count

                if self.config.restrict_edit_to_layers == []:
                    self.config.restrict_edit_to_layers = list(
                        range(self.config.n_layer)
                    )
                if self.config.restrict_edit_to_positions == []:
                    self.config.restrict_edit_to_positions = list(
                        range(encoder_hidden_states.shape[1] // self.config.n_layer)
                    )

                all_layers = set(
                    range(self.config.n_layer)
                )  # Create a set of numbers from 0 to 12
                edit_layers = set(
                    self.config.restrict_edit_to_layers
                )  # Convert restrict_edit_to_layers to a set
                all_positions = set(
                    range(encoder_hidden_states.shape[1] // self.config.n_layer)
                )
                edit_positions = set(self.config.restrict_edit_to_positions)
                layers_to_omit = all_layers - edit_layers
                positions_to_omit = all_positions - edit_positions

                for layer in layers_to_omit:
                    mask[:, :, layer] = torch.zeros_like(mask[:, :, layer])

                for position in positions_to_omit:
                    mask[:, position, :] = torch.zeros_like(mask[:, position, :])

                # Havne't checked, but this next line should be effectively stacking the mask along d=2
                mask = mask.reshape(encoder_hidden_states.shape[0], -1).to(
                    hidden_states.device
                )

                if encoder_attention_mask is not None:
                    encoder_attention_mask = (encoder_attention_mask * mask).to(
                        hidden_states.device
                    )
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
                    stacking_dim = 1
                    outputs += (attn_weights.unsqueeze(stacking_dim),)
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
                    # stacking_dim = attn_weights.shape.index(self.heads_per_multiply)

                    attn_output_old, _, attn_weights_old = outputs
                    attn_output += attn_output_old

                    attn_weights = torch.cat(
                        (attn_weights_old, attn_weights.unsqueeze(1)), dim=stacking_dim
                    )
                    outputs = (attn_output, present, attn_weights)
                else:
                    attn_output_old, _ = outputs
                    attn_output += attn_output_old
                    outputs = (attn_output, present)

        return outputs  # a, present, (attentions)
