from __future__ import annotations

from typing import Any, Callable, List, Mapping, Tuple, TypeVar

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
)

from .utils import (
    EditorConfig,
    EditorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)

T = TypeVar("T", bound="BaseEditor")


class BaseEditor(nn.Module):
    def __init__(self, config: EditorConfig):
        super().__init__()
        self.config = config
        if hasattr(config, "n_embd"):
            self.n_embd = config.n_embd
        if hasattr(config, "n_layer"):
            self.n_layer = config.n_layer
        self._post_initialization(config)

    def _post_initialization(self, config: EditorConfig):
        self.set_hypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config._name_or_path
        ).eval()

        # freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        assign_layer_indices(self.target_model)

        if config.use_layerwise_embeddings:
            # extra layer is cross-attn in the lm_head
            self.layerwise_embeddings = nn.Parameter(
                torch.zeros(self.n_layer + 1, self.n_embd), requires_grad=True
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

    def get_edit_hooks(
        self, batch_edit_vectors: torch.Tensor
    ) -> Tuple[Callable, Callable]:
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

        return edit_add, embedding_edit_add

    def get_hooks(
        self, batch_edit_vectors: torch.Tensor
    ) -> List[Tuple[nn.Module, Callable]]:
        raise NotImplementedError

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
                    self.n_layer + 1,
                    self.n_embd,
                )
            )

        # If we are stopping editing at stop_editing_index,
        # this pads batch_edit_vectors with 0's to the left of the edited positions
        if stop_editing_idx is not None:
            padded_batch_edits = torch.zeros(
                batch_edit_vectors.shape[0],
                target_input_ids.shape[1],
                self.n_layer + 1,
                self.n_embd,
                device=batch_edit_vectors.device,
                dtype=batch_edit_vectors.dtype,
            )
            padded_batch_edits[stop_edit_mask] = batch_edit_vectors.reshape(
                -1, *batch_edit_vectors.shape[2:]
            )
            batch_edit_vectors = padded_batch_edits

        hooks = self.get_hooks(batch_edit_vectors)

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
