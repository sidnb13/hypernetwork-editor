from transformers import (
    GPT2Config,
)

from ..utils import (
    EditorConfig,
)


class GPT2EditorConfig(GPT2Config, EditorConfig):
    init_attn_proj_bias: bool = True
    compute_position_ids: bool = True
