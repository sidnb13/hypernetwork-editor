import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from logger import get_logger

from .config import Phi3EditorConfig

logger = get_logger(__name__)


class Phi3EditorUnembedCrossAttention(Phi3Attention):
    is_cross_attention = True

    def __init__(self, config: Phi3EditorConfig, layer_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size

        self.num_heads = config.num_editing_heads
        self.head_dim = (
            self.hidden_size * config.edit_channel_multiply_factor // self.num_heads
        )
        # NOTE: do these stay the same?
        self.num_key_value_heads = config.num_editing_heads
        self.num_key_value_groups = (
            self.num_heads
            * config.edit_channel_multiply_factor
            // self.num_key_value_heads
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (
            self.head_dim * self.num_heads
        ) != self.hidden_size * config.edit_channel_multiply_factor:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.ModuleList(
            [
                nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
                for _ in range(config.edit_channel_multiply_factor)
            ]
        )
        self.q_proj = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(config.edit_channel_multiply_factor)
            ]
        )
        self.kv_proj = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, op_size, bias=False)
                for _ in range(config.edit_channel_multiply_factor)
            ]
        )

        self._init_rope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if encoder_hidden_states is None:
            raise ValueError(
                "If class is used as cross-attention, the encoder_hidden_states cannot be None"
            )

        logger.warning_once(
            "You are not running the flash-attention implementation, expect numerical differences."
        )

        bsz, q_len, _ = hidden_states.size()

        for i in range(self.config.edit_channel_multiply_factor):
            kv = self.kv_proj[i](hidden_states)
            query_states = self.q_proj[i](encoder_hidden_states)
            key_states = kv[..., : self.num_key_value_heads * self.head_dim]
            value_states = kv[..., self.num_key_value_heads * self.head_dim :]

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }  # Specific to RoPE models
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights += causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(value_states.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )

            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            attn_output = self.o_proj[i](attn_output)

            if not output_attentions:
                attn_weights = None

            if i == 0:
                outputs = (attn_output, None)
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
                    outputs = (attn_output, None, attn_weights)
                else:
                    attn_output_old, _ = outputs
                    attn_output += attn_output_old
                    outputs = (attn_output, None)

        return outputs


class Phi3CrossAttention(Phi3Attention):
    """PhiAttention for the unembed layer with support for cross-attention."""

    is_cross_attention = True

    def __init__(self, config: Phi3EditorConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if encoder_hidden_states is None:
            raise ValueError(
                "If class is used as cross-attention, the encoder_hidden_states cannot be None"
            )

        logger.warning_once(
            "You are not running the flash-attention implementation, expect numerical differences."
        )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(value_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
