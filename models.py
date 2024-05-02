from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2FlashAttention2
from transformers.pytorch_utils import Conv1D


class GPT2EditorConfig(GPT2Config):
    edit_channel_width: int = 2
    chop_editor_at_layer: int = 12
    num_editing_heads: int = 32
    use_layerwise_embeddings: bool = True
    edit_dampening_factor: float = 0.001
    kill_token_zero: bool = False


class EditorCrossAttention(GPT2Attention):
    is_cross_attention = True

    def __init__(self, config: GPT2EditorConfig, layer_idx=None, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_editing_heads

        assert (
            self.num_heads % config.edit_channel_width == 0
        ), f"num_editing_heads ({config.num_editing_heads}) must be divisible by edit_channel_width ({config.edit_channel_width})"

        self.head_dim = self.embed_dim * config.edit_channel_width // self.num_heads

        # split additional factor of channel width
        self.split_size = self.embed_dim * self.config.edit_channel_width

        if (
            self.head_dim * self.num_heads
            != self.embed_dim * self.config.edit_channel_width
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
            2 * self.embed_dim * self.config.edit_channel_width, self.embed_dim
        )
        self.q_attn = Conv1D(
            self.embed_dim * self.config.edit_channel_width, self.embed_dim
        )
        self.c_proj = Conv1D(
            self.embed_dim, self.embed_dim * self.config.edit_channel_width
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
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

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
            query = self.q_attn(hidden_states)
            
            # We only take the last position hidden state from the editor
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

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

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class EditorCrossFlashAttention2(EditorCrossAttention, GPT2FlashAttention2):
    pass


class GPT2Editor(GPT2LMHeadModel):
    _tied_weights_keys = []

    def __init__(self, config: GPT2EditorConfig):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        if config._attn_implementation == "flash_attention_2":
            self.lm_head = EditorCrossFlashAttention2(
                config=config, layer_idx=config.chop_editor_at_layer
            )
        else:
            self.lm_head = EditorCrossAttention(
                config=config, layer_idx=config.chop_editor_at_layer
            )

        # prune layers
        self.transformer.layers = self.transformer.layers[: config.chop_editor_at_layer]
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

    def new_forward(
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

        # lm_logits = self.lm_head(hidden_states)
        reverse_attention_output = self.lm_head(
            hidden_states, encoder_hidden_states, output_attentions=output_attentions
        )

        return reverse_attention_output
