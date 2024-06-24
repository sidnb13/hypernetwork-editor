from transformers import Phi3Config

from ..utils import EditorConfig


class Phi3EditorConfig(Phi3Config, EditorConfig):
    init_attn_proj_bias: bool = True
    compute_position_ids: bool = True
