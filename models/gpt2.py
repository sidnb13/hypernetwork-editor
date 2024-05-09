from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2FlashAttention2,
)

from models.utils import EditorModelOutput

from .utils import EditorConfig, add_fwd_hooks, assign_layer_indices


class GPT2EditorConfig(GPT2Config, EditorConfig):
    init_attn_proj_bias: bool = False


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
        #print device of all components
        # print(self.bias.device)
        # print(x.view(-1, x.size(-1)).device)
        # print(self.weight.device)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


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
            return (attn_output, attn_weights)
        else:
            return (attn_output,)


class EditorUnembedCrossAttention(nn.Module):
    is_cross_attention = True

    def __init__(self, config: GPT2EditorConfig, layer_idx=None, **kwargs):
        super().__init__()
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

        #Note by Mike: so, how to add more expressivity going forward?
        #I think the natural thing to do is to add support for many more heads, by changing the
        # matrix-mulitply code below, such that it effectively processes only d_embed of the attention channel at a time
        # in a loop, adding the results of that part of the attention computation to the output tensor
        # This will allow us to use much wider edit_channel_multiply_factor!

        assert (
            self.num_heads % config.edit_channel_multiply_factor == 0
        ), f"num_editing_heads ({config.num_editing_heads}) must be divisible by edit_channel_width ({config.edit_channel_multiply_factor})"

        self.heads_per_multiply = self.num_heads // config.edit_channel_multiply_factor

        self.head_dim = (
            self.embed_dim * config.edit_channel_multiply_factor // self.num_heads
        )

        # # split additional factor of channel width
        # # Changing this back to embed_dim, now that we're accumulating multiplies!
        self.split_size = self.embed_dim #* self.config.edit_channel_multiply_factor

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

        self.c_attn = []
        self.q_attn = []
        self.c_proj = []

        for i in range(config.edit_channel_multiply_factor):
            self.c_attn.append(
                Conv1D(
                    2 * self.embed_dim,
                    self.embed_dim,
                    init_bias=config.init_attn_proj_bias,
                )
            )
            self.q_attn.append(
                Conv1D(
                    self.embed_dim,
                    self.embed_dim,
                    init_bias=config.init_attn_proj_bias,
                )
            )
            self.c_proj.append(
                Conv1D(
                    self.embed_dim,
                    self.embed_dim,
                    init_bias=config.init_attn_proj_bias,
                )
            )
        # # edit channel width specifies
        # self.c_attn = Conv1D(
        #     2 * self.embed_dim * self.config.edit_channel_multiply_factor,
        #     self.embed_dim,
        #     init_bias=config.init_attn_proj_bias,
        # )
        # self.q_attn = Conv1D(
        #     self.embed_dim * self.config.edit_channel_multiply_factor,
        #     self.embed_dim,
        #     init_bias=config.init_attn_proj_bias,
        # )
        # self.c_proj = Conv1D(
        #     self.embed_dim,
        #     self.embed_dim * self.config.edit_channel_multiply_factor,
        #     init_bias=config.init_attn_proj_bias,
        # )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

        # take methods from GPT2Attention and GPT2FlashAttention implementations
        self._flash_attn_enabled = config._attn_implementation == "flash_attention_2"
        self._attn = (
            GPT2FlashAttention2._flash_attention_forward.__get__(
                self, GPT2FlashAttention2
            )
            if self._flash_attn_enabled
            else GPT2Attention._attn.__get__(self, GPT2Attention)
        )
        self._merge_heads = GPT2Attention._merge_heads.__get__(self, GPT2Attention)
        self._upcast_and_reordered_attn = (
            GPT2Attention._upcast_and_reordered_attn.__get__(self, GPT2Attention)
        )

    def _split_heads(self, tensor):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (self.heads_per_multiply, self.head_dim)
        return tensor.view(new_shape)

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
            #throw an error
            print("Error: This class is only meant to be used as cross attention")
            #query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        for i in range(self.config.edit_channel_multiply_factor):
            
            query = self.q_attn[i](encoder_hidden_states)
            # We only take the last position hidden state from the editor
            key, value = self.c_attn[i](hidden_states[:, -1, :]).split(
                self.split_size, dim=-1
            )
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

            if self._flash_attn_enabled:
                attn_dropout = self.attn_dropout.p if self.training else 0.0
                attn_output = self._attn(
                    query, key, value, attention_mask, query_len, dropout=attn_dropout
                )
                attn_weights = attn_output.reshape(
                    bsz, query_len, self.num_heads * self.head_dim
                )
            else:
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
            attn_output = self._merge_heads(attn_output, self.heads_per_multiply, self.head_dim)
            attn_output = self.c_proj[i](attn_output)
            attn_output = self.resid_dropout(attn_output)

            if i == 0:
                outputs = (attn_output, present)
                if output_attentions:
                    outputs += (attn_weights,)
            else:
                #print(outputs[0][0])
                outputs[0][0] += attn_output[0]
                if use_cache is True:
                    print("Error, key-value caching for this is not implemented. Should we even be doing this? -Mike")
                    #outputs[1] = ( torch.stack( (outputs[1][0], present[0]) ), torch.stack( (outputs[1][1] , present[1]))) #genuinely unsure what the stacking axes should be!
                if output_attentions:
                    #Check stacking dimensions! 
                    #Find which dimension of attn_weights is equal to the number of heads per multiply
                    #Then stack along that dimension
                    #Don't use number of heads equal to 786 until this is cleared up!
                    stacking_dim = torch.argmax(attn_weights.shape == self.heads_per_multiply)
                    print("stacking dimension is")
                    print(stacking_dim)
                    outputs[2] = torch.stack((outputs[2], attn_weights), dim=stacking_dim)

        return outputs  # a, present, (attentions)


class GPT2EditorHypernetwork(GPT2LMHeadModel):
    _tied_weights_keys = []

    def __init__(self, config: GPT2EditorConfig):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # only LM head gets special attention
        if config._attn_implementation == "flash_attention_2":
            _attention_cls = GPT2FlashAttention2
        else:
            _attention_cls = GPT2Attention

        self.lm_head = EditorUnembedCrossAttention(
            config=config, layer_idx=config.chop_editor_at_layer
        )
        # self.lm_head = OldEditorAttention(config)

        # prune layers and add cross attn heads
        self.transformer.h = self.transformer.h[: config.chop_editor_at_layer]
        for i, layer in enumerate(self.transformer.h):
            layer.crossattention = _attention_cls(
                config=config, layer_idx=i, is_cross_attention=True
            )
            layer.ln_cross_attn = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_epsilon
            )

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        #set device for input_ids to cuda ?
        #input_ids = input_ids.to(self.lm_head.weight.device)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        # else: #delete?  #IDK if this next pair of lines is necessary at all!
        #     hidden_states = hidden_states.to(self.lm_head.weight.device) # delete?

        reverse_attention_output = self.lm_head(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
        )

        # (output, present[,attentions])
        return reverse_attention_output


class GPT2Editor(nn.Module):
    def __init__(self, config: GPT2EditorConfig):
        super().__init__()

        self.config = config
        self.hypernetwork = GPT2EditorHypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(config.name_or_path)
        assign_layer_indices(self.target_model)

        if config.use_layerwise_embeddings:
            # extra layer is cross-attn in the lm_head
            self.layerwise_embeddings = nn.Parameter(
                torch.zeros(config.n_layer + 1, config.n_embd), requires_grad=True
            )
            self.layerwise_embeddings.data.normal_(
                mean=0.0, std=self.target_model.config.initializer_range
            )
        else:
            self.layerwise_embeddings = None

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self, target_input_ids: torch.Tensor, target_attention_mask: torch.Tensor
    ):
        """Gets the hidden states from the target model, if necessary"""
        outputs = self.target_model(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        return hidden_states

    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        editor_attention_mask: torch.Tensor = None,
        target_input_ids: torch.Tensor = None,
        target_attention_mask: torch.Tensor = None,
        target_hidden_states: torch.Tensor = None,
        output_target_hidden_states: bool = False,
        output_edited_hidden_states: bool = False,
        output_edit_vectors: bool = False,
        output_editor_attention: bool = False,
        stop_editing_idx: int = None,
        batch_edit_vectors: torch.Tensor = None,
    ) -> EditorModelOutput:
        # Run target model for encoded hidden states
        if target_hidden_states is None:
            target_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    target_input_ids, target_attention_mask
                ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                dim=2,
            )
        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768

        # If we are stopping editing at stop_editing_idx, then we eliminate target_hidden_states beyond that index
        if stop_editing_idx is not None:
            target_hidden_states = target_hidden_states[
                :, :stop_editing_idx, :, :
            ].clone()

        # Normalize along the last dimension
        normalization_factors = target_hidden_states.norm(dim=-1, keepdim=True)
        target_hidden_states = target_hidden_states / normalization_factors

        # Error catching:
        if batch_edit_vectors is not None:
            if output_edit_vectors or output_editor_attention:
                raise ValueError(
                    "Inputting your own batch_edit_vectors means the model does not construct the outputs you are requesting"
                )

        # Run editor model, get edit vectors
        if batch_edit_vectors is None:
            if self.layerwise_embeddings is not None:
                # Now, add in the layerwise embeddings
                embedded_hidden_states = (
                    target_hidden_states + self.layerwise_embeddings[None, None, :, :]
                )

                collapsed_target_hidden_states = embedded_hidden_states.reshape(
                    target_hidden_states.shape[0],
                    target_hidden_states.shape[1] * target_hidden_states.shape[2],
                    target_hidden_states.shape[3],
                )
            else:
                collapsed_target_hidden_states = target_hidden_states.reshape(
                    target_hidden_states.shape[0],
                    target_hidden_states.shape[1] * target_hidden_states.shape[2],
                    target_hidden_states.shape[3],
                )

            editor_output = self.hypernetwork(
                input_ids=editor_input_ids,
                attention_mask=editor_attention_mask,
                encoder_hidden_states=collapsed_target_hidden_states,
                output_attentions=output_editor_attention,
            )

            # Multiply the outputs by normalization factors
            if output_editor_attention:
                temp_edit_vectors, batch_editor_attention = editor_output
            else:
                temp_edit_vectors = editor_output[0]

            # Renormalize to the scale of the target hidden states
            # and reshape to proper dimensions
            batch_edit_vectors = (
                self.config.edit_dampening_factor
                * normalization_factors
                * temp_edit_vectors.reshape(
                    temp_edit_vectors.shape[0],
                    -1,
                    self.config.n_layer + 1,
                    self.config.n_embd,
                )
            )

        # If we are stopping editing at stop_editing_index,
        # this pads batch_edit_vectors with 0's to the right of the edited positions
        if stop_editing_idx is not None:
            batch_edit_vectors = torch.cat(
                (
                    batch_edit_vectors,
                    torch.zeros(
                        batch_edit_vectors.shape[0],
                        target_input_ids.shape[1] - stop_editing_idx,
                        self.config.n_layer + 1,
                        self.config.n_embd,
                    ),
                ),
                dim=1,
            )

        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def edit_add(module, input, output):
            layer_index = module.layer_index
            output[0][:] = output[0] + batch_edit_vectors[:, :, layer_index, :]
            if self.config.kill_token_zero:
                output[0][:, 0, :] = 0

        def embedding_edit_add(module, input, output):
            output[:] = output + batch_edit_vectors[:, :, 0, :]
            if self.config.kill_token_zero:
                output[:, 0, :] = 0

        # Now editing the target model
        hooks1 = [(self.target_model.transformer.wte, embedding_edit_add)]
        hooks2 = [
            (self.target_model.transformer.h[L], edit_add)
            for L in range(self.target_model.config.n_layer)
        ]
        hooks = hooks1 + hooks2
        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                output_hidden_states=output_edited_hidden_states,
            )

        logits = target_result.logits

        output = EditorModelOutput(logits=logits)
        if output_target_hidden_states:
            output.target_hidden_states = target_hidden_states * normalization_factors
        if output_edited_hidden_states:
            output.edited_hidden_states = target_result.hidden_states
        if output_edit_vectors:
            output.edit_vectors = batch_edit_vectors
        if output_editor_attention:
            output.editor_attention = batch_editor_attention
        return output
