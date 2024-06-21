from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union

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

from ..utils import (
    EditorConfig,
    EditorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)
from .layers import EditorUnembedCrossAttention

T = TypeVar("T", bound="GPT2Editor")


class GPT2EditorConfig(GPT2Config, EditorConfig):
    init_attn_proj_bias: bool = False
    compute_position_ids: bool = True
    use_ghost_token: bool = False
    cross_attn_layers: List[int] = []
    restrict_edit_to_layers: List[int] = []
    restrict_edit_to_positions: List[int] = []

class CustomIdentity(nn.Identity):
    def forward(self, input, *args, **kwargs):
        return input

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

        # prune layers and add cross attn heads
        if config.chop_editor_at_layer == 0:
            self.transformer.h = nn.ModuleList([CustomIdentity()])
        else:
            self.transformer.h = self.transformer.h[: config.chop_editor_at_layer]

        if config.cross_attn_layers == []:
            config.cross_attn_layers = list(range(config.chop_editor_at_layer))

        if config.chop_editor_at_layer != 0:
            for i, layer in enumerate(self.transformer.h):
                ### if i not in config.cross_attn_layers:
                ###     continue
                layer.crossattention = _attention_cls(
                    config=config, layer_idx=i, is_cross_attention=True
                )
                layer.ln_cross_attn = nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_epsilon
                )
                original_query_weights = layer.attn.c_attn.weight[:, : config.hidden_size]
                original_keys_values = layer.attn.c_attn.weight[:, config.hidden_size :]
                original_query_bias = layer.attn.c_attn.bias[: config.hidden_size]
                original_keys_values_bias = layer.attn.c_attn.bias[config.hidden_size :]

                # with torch.no_grad():
                # Initialize the new layer with these parameters
                # just added calls to .detach()
                layer.crossattention.q_attn.weight = nn.Parameter(original_query_weights.detach())
                layer.crossattention.q_attn.bias = nn.Parameter(original_query_bias.detach())
                layer.crossattention.c_attn.weight = nn.Parameter(original_keys_values.detach())
                layer.crossattention.c_attn.bias = nn.Parameter(original_keys_values_bias.detach())

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

                #currently not working? some gradients are passing thru

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


class GPT2Editor(nn.Module):
    def __init__(self, config: GPT2EditorConfig):
        super().__init__()

        self.config = config
        self.hypernetwork = GPT2EditorHypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path
        ).eval()

        # freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

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

    def train(self: T, mode: bool = True) -> T:
        return self.hypernetwork.train(mode)

    def eval(self: T) -> T:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        """Gets the hidden states from the target model, if necessary"""

        if position_ids is not None:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        else:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                output_hidden_states=True,
            )

        return outputs.hidden_states

    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        editor_attention_mask: torch.Tensor = None,
        target_input_ids: torch.Tensor = None,
        target_attention_mask: torch.Tensor = None,
        target_hidden_states: torch.Tensor = None,
        target_position_ids: torch.Tensor = None,
        output_target_hidden_states: bool = False,
        output_edited_hidden_states: bool = False,
        output_edit_vectors: bool = False,
        output_editor_attention: bool = False,
        stop_editing_idx: int = None,
        batch_edit_vectors: torch.Tensor = None,
    ) -> EditorModelOutput:
        # Run target model for encoded hidden states
        if target_hidden_states is None:
            # Add a ghost token to the input_ids and turn off its attention mask
            if self.config.use_ghost_token:
                ghost_token = torch.zeros_like(target_input_ids[:, 0:1])
                target_input_ids = torch.cat((ghost_token, target_input_ids), dim=1)
                ghost_invisible_attention_mask = torch.cat(
                    (torch.zeros_like(ghost_token), target_attention_mask), dim=1
                )
                ghost_present_attention_mask = torch.cat(
                    (torch.ones_like(ghost_token), target_attention_mask), dim=1
                )

                # This is altered! We give a position_id of 42 to the ghost token by default.
                if target_position_ids is None:
                    # create position_ids on the fly for batch generation
                    target_position_ids = (
                        ghost_invisible_attention_mask.long().cumsum(-1) - 1
                    )
                    target_position_ids.masked_fill_(
                        ghost_invisible_attention_mask == 0, 1
                    )
                    target_position_ids[:, 0] = torch.full_like(
                        target_position_ids[:, 0], 42
                    )
                    # we aren't using past_key_values I assume. If we did, we'd use something like:
                    # if past_key_values:
                    #     position_ids = position_ids[:, -input_ids.shape[1] :]

                target_hidden_states = torch.stack(
                    self._run_target_model_for_encoded_hidden_states(
                        target_input_ids,
                        target_attention_mask=ghost_invisible_attention_mask,
                        position_ids=target_position_ids,
                    ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                    dim=2,
                )
            else:
                target_hidden_states = torch.stack(
                    self._run_target_model_for_encoded_hidden_states(
                        target_input_ids, target_attention_mask, target_position_ids
                    ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                    dim=2,
                )
        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768

        if self.config.use_ghost_token:
            target_attention_mask = ghost_present_attention_mask

        mask_sum = target_attention_mask.cumsum(-1)
        mask_sum_min = mask_sum.min(dim=-1)[0]
        edit_window_mask = (
            torch.arange(
                0, target_attention_mask.shape[-1], device=target_attention_mask.device
            )
            .unsqueeze(0)
            .repeat(target_attention_mask.shape[0], 1)
        )
        edit_window_mask[mask_sum > 0] += (
            mask_sum_min.unsqueeze(-1).repeat(1, mask_sum.shape[-1]).view(-1)
        )
        stop_edit_mask = torch.logical_and(
            edit_window_mask > 0, edit_window_mask <= stop_editing_idx
        )

        # If we are stopping editing at stop_editing_idx, then we eliminate target_hidden_states beyond that index
        if stop_editing_idx is not None:
            target_hidden_states = (
                target_hidden_states[stop_edit_mask]
                .reshape(
                    target_hidden_states.shape[0],
                    stop_editing_idx,
                    *target_hidden_states.shape[2:],
                )
                .clone()
            )

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
                temp_edit_vectors, _, batch_editor_attention = editor_output
            else:
                temp_edit_vectors, _ = editor_output

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
        # this pads batch_edit_vectors with 0's to the left of the edited positions
        if stop_editing_idx is not None:
            padded_batch_edits = torch.zeros(
                batch_edit_vectors.shape[0],
                target_input_ids.shape[1],
                self.config.n_layer + 1,
                self.config.n_embd,
                device=batch_edit_vectors.device,
                dtype=batch_edit_vectors.dtype,
            )
            padded_batch_edits[stop_edit_mask] = batch_edit_vectors.reshape(
                -1, *batch_edit_vectors.shape[2:]
            )
            batch_edit_vectors = padded_batch_edits

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

        if self.config.use_ghost_token:
            target_attention_mask = ghost_present_attention_mask

        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                position_ids=target_position_ids,
                output_hidden_states=output_edited_hidden_states,
            )

        # Drop the outputs atop the ghost token
        if self.config.use_ghost_token:
            target_result.logits = target_result.logits[:, 1:, :]

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
