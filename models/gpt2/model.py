from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Model,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2FlashAttention2,
)

from models.base import BaseEditor

from ..utils import (
    CustomIdentity,
)
from .config import GPT2EditorConfig
from .layers import GPT2EditorUnembedCrossAttention


class GPT2EditorHypernetwork(GPT2LMHeadModel):
    _tied_weights_keys = []

    def __init__(self, config: GPT2EditorConfig):
        nn.Module.__init__(self)
        self.config = config
        self.transformer = GPT2Model(config)
        # only LM head gets special attention
        if config._attn_implementation == "flash_attention_2":
            _attention_cls = GPT2FlashAttention2
        else:
            _attention_cls = GPT2Attention

        self.lm_head = GPT2EditorUnembedCrossAttention(
            config=config, layer_idx=config.chop_editor_at_layer
        )

        # prune layers and add cross attn heads
        if config.chop_editor_at_layer == 0:
            self.transformer.h = nn.ModuleList([CustomIdentity()])
        else:
            self.transformer.h = self.transformer.h[: config.chop_editor_at_layer]

        if config.cross_attn_layers == []:
            config.cross_attn_layers = list(range(config.chop_editor_at_layer))

        if config.chop_editor_at_layer > 0:
            for i, layer in enumerate(self.transformer.h):
                ### if i not in config.cross_attn_layers:
                ###     continue
                layer.crossattention = _attention_cls(
                    config=config, layer_idx=i, is_cross_attention=True
                )
                layer.ln_cross_attn = nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_epsilon
                )
                original_query_weights = layer.attn.c_attn.weight[
                    :, : config.hidden_size
                ]
                original_keys_values = layer.attn.c_attn.weight[:, config.hidden_size :]
                original_query_bias = layer.attn.c_attn.bias[: config.hidden_size]
                original_keys_values_bias = layer.attn.c_attn.bias[config.hidden_size :]

                # with torch.no_grad():
                # Initialize the new layer with these parameters
                # just added calls to .detach()
                layer.crossattention.q_attn.weight = nn.Parameter(
                    original_query_weights.detach()
                )
                layer.crossattention.q_attn.bias = nn.Parameter(
                    original_query_bias.detach()
                )
                layer.crossattention.c_attn.weight = nn.Parameter(
                    original_keys_values.detach()
                )
                layer.crossattention.c_attn.bias = nn.Parameter(
                    original_keys_values_bias.detach()
                )

                ### Jankiness: I think this is a way of shutting everything off?
                # #if i not in config.cross_attn_layers:
                #     #set them all to zero and kill gradient
                # layer.crossattention.q_attn.weight.data.zero_()
                # layer.crossattention.q_attn.bias.data.zero_()
                # layer.crossattention.c_attn.weight.data.zero_()
                # layer.crossattention.c_attn.bias.data.zero_()
                # layer.crossattention.c_proj.weight.data.zero_()
                # layer.crossattention.c_proj.bias.data.zero_()

                # layer.crossattention.q_attn.weight.requires_grad = False
                # layer.crossattention.q_attn.bias.requires_grad = False
                # layer.crossattention.c_attn.weight.requires_grad = False
                # layer.crossattention.c_attn.bias.requires_grad = False
                # layer.crossattention.c_proj.weight.requires_grad = False
                # layer.crossattention.c_proj.bias.requires_grad = False

                # currently not working? some gradients are passing thru

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
        # set device for input_ids to cuda ?
        # input_ids = input_ids.to(self.lm_head.weight.device)
        if (
            attention_mask is not None
            and position_ids is None
            and self.config.compute_position_ids
        ):
            position_ids = attention_mask.cumsum(-1)

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
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
        )

        # (output, present[,attentions])
        return reverse_attention_output


class GPT2Editor(BaseEditor):
    def __init__(self, config: GPT2EditorConfig):
        super().__init__(config)

    def set_hypernetwork(self, config: GPT2EditorConfig):
        self.hypernetwork = GPT2EditorHypernetwork(config)

    def get_hooks(self, batch_edit_vectors):
        edit_add, embedding_edit_add = self.get_edit_hooks(batch_edit_vectors)
        # Now editing the target model
        hooks1 = [(self.target_model.transformer.wte, embedding_edit_add)]
        hooks2 = [
            (self.target_model.transformer.h[L], edit_add)
            for L in range(self.target_model.config.n_layer)
        ]
        hooks = hooks1 + hooks2

        return hooks
