from typing import List

from transformers import GPT2Config

from ..utils import (
    EditorConfig,
)


class GPT2EditorConfig(GPT2Config, EditorConfig):
    init_attn_proj_bias: bool = False
    compute_position_ids: bool = True
    use_ghost_token: bool = False
    cross_attn_layers: List[int] = []
    restrict_edit_to_layers: List[int] = []
    restrict_edit_to_positions: List[int] = []
