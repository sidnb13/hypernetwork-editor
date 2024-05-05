import os
from typing import Dict

import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_tokenizer(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token_id = tok.eos_token_id
    return tok


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {
        k: (v.to(rank) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()
    }
    return on_device
