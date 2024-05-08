import os
from typing import Dict

import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_tokenizer(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token_id = tok.eos_token_id
    # This is very important because we take last hidden state in editor
    tok.padding_side = "left"
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


def concat_and_pad_ids(batch: dict, pad_token: int):
    first, second = batch["editor_input_ids"], batch["target_input_ids"]
    batch_size, first_seq = first.size()
    _, second_seq = second.size()
    # Find the lengths in A and B
    lengths_A = torch.sum(batch["editor_attention_mask"] > 0, dim=1)
    lengths_B = torch.sum(batch["target_attention_mask"] > 0, dim=1)
    # initialize empty tensor
    max_len = max(lengths_A + lengths_B)
    result = torch.full(
        (
            batch_size,
            max_len,
        ),
        pad_token,
        device=first.device,
        dtype=first.dtype,
    )
    # Concatenate A[i] and B[i] a, assume LEFT padding
    for i in range(batch_size):
        result[i, max_len - lengths_B[i] - lengths_A[i] : max_len - lengths_B[i]] = (
            first[i, first_seq - lengths_A[i] :]
        )
        result[i, max_len - lengths_B[i] :] = second[i, second_seq - lengths_B[i] :]

    return result
