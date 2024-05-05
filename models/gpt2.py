from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    PreTrainedModel,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2FlashAttention2,
)
from transformers.pytorch_utils import Conv1D

from models.utils import EditorModelOutput

from .utils import EditorConfig, add_fwd_hooks, assign_layer_indices


class GPT2EditorConfig(GPT2Config, EditorConfig):
    pass


class EditorCrossAttention(GPT2Attention):
    is_cross_attention = True

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

        assert (
            self.num_heads % config.edit_channel_width_factor == 0
        ), f"num_editing_heads ({config.num_editing_heads}) must be divisible by edit_channel_width ({config.edit_channel_width_factor})"

        self.head_dim = (
            self.embed_dim * config.edit_channel_width_factor // self.num_heads
        )

        # split additional factor of channel width
        self.split_size = self.embed_dim * self.config.edit_channel_width_factor

        if (
            self.head_dim * self.num_heads
            != self.embed_dim * self.config.edit_channel_width_factor
        ):
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # edit channel width specifies
        self.c_attn = Conv1D(
            2 * self.embed_dim * self.config.edit_channel_width_factor, self.embed_dim
        )
        self.q_attn = Conv1D(
            self.embed_dim * self.config.edit_channel_width_factor, self.embed_dim
        )
        self.c_proj = Conv1D(
            self.embed_dim, self.embed_dim * self.config.edit_channel_width_factor
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    def _split_heads(self, tensor):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
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
            query = self.q_attn(encoder_hidden_states)
            # We only take the last position hidden state from the editor
            key, value = self.c_attn(hidden_states[:, -1, :]).split(
                self.split_size, dim=-1
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        split_query = self._split_heads(query)
        bsz, _, num_heads, head_dim = split_query.size()

        # (bsz, seq_len, num_heads, head_dim) -> (bsz * num_heads, seq_len, head_dim)
        query = split_query.permute(0, 2, 1, 3).reshape(bsz * num_heads, -1, head_dim)
        key = (
            self._split_heads(key).unsqueeze(-2).reshape(bsz * num_heads, -1, head_dim)
        )
        value = (
            self._split_heads(value)
            .unsqueeze(-2)
            .reshape(bsz * num_heads, -1, head_dim)
        )

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
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class EditorCrossFlashAttention2(EditorCrossAttention, GPT2FlashAttention2):
    pass


class GPT2EditorHypernetwork(GPT2LMHeadModel):
    _tied_weights_keys = []

    def __init__(self, config: GPT2EditorConfig):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # only LM head gets special attention
        if config._attn_implementation == "flash_attention_2":
            self.lm_head = EditorCrossFlashAttention2(
                config=config, layer_idx=config.chop_editor_at_layer
            )
            _attention_cls = GPT2FlashAttention2
        else:
            self.lm_head = EditorCrossAttention(
                config=config, layer_idx=config.chop_editor_at_layer
            )
            _attention_cls = GPT2Attention

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

        reverse_attention_output = self.lm_head(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
        )

        # (output, present[,attentions])
        return reverse_attention_output


class GPT2Editor(PreTrainedModel):
    def __init__(self, config: GPT2EditorConfig):
        super().__init__(config)
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
        stop_editing_index: int = None,
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

        # If we are stopping editing at stop_editing_index, then we eliminate target_hidden_states beyond that index
        if stop_editing_index is not None:
            target_hidden_states = target_hidden_states[
                :, :stop_editing_index, :, :
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
        if stop_editing_index is not None:
            batch_edit_vectors = torch.cat(
                (
                    batch_edit_vectors,
                    torch.zeros(
                        batch_edit_vectors.shape[0],
                        target_input_ids.shape[1] - stop_editing_index,
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
        hooks2 = [(self.target_model.transformer.h[L], edit_add) for L in range(12)]
        hooks = hooks1 + hooks2
        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                target_input_ids,
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
