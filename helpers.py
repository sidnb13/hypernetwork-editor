import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from models.utils import EditorModelOutput

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def visualize_attn_heatmap(
    result: EditorModelOutput,
    batch: Dict,
    save_path: str | os.PathLike = None,
    show_plot: bool = False,
    tokenizer: AutoTokenizer = None,
    stopping_index: int = None,
    metadata: DictConfig = None,
    step: int = None,
):
    if save_path:
        if step is not None:
            save_path = Path(save_path) / "viz-step-{}".format(step)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(save_path) / f"viz-{timestamp}"
        show_plot = False
        os.makedirs(save_path, exist_ok=True)

        if metadata:
            OmegaConf.save(metadata, save_path / "config.yaml")

    for batch_index in range(len(next(iter(batch.values())))):
        # The tensor norm comes in an stopping_indexnum_layers+1 matrix
        edit_tensor = result.edit_vectors[batch_index].cpu()
        target_hidden = result.target_hidden_states[batch_index].cpu()

        edit_tensor[:stopping_index, :, :] = edit_tensor[
            :stopping_index, :, :
        ] / target_hidden[:stopping_index].norm(dim=2, keepdim=True)
        edit_tensor_norm = edit_tensor.norm(dim=2).flip(1)

        # is this any better??
        # attention_matrix = result['editor_attention'][batch_index].reshape(104).to("cpu").reshape(13,8).permute(1,0)

        # Detach and convert to numpy
        edit_tensor_norm = edit_tensor_norm.numpy()[:stopping_index, :]

        # Create the heatmap
        fig, ax = plt.subplots()
        heatmap = ax.imshow(edit_tensor_norm.transpose(), cmap="hot")

        # Color the heatmap according to the entry sizes
        heatmap.set_clim(vmin=np.min(0), vmax=np.max(edit_tensor_norm))
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Entry Sizes")

        # TODO: pass model config prevent hardcoding
        # Add labels to the x and y axes
        ax.set_yticks(np.arange(13))
        ax.set_xticks(np.arange(8))
        # ax.set_yticklabels(np.arange(13))
        ax.set_yticklabels(np.arange(12, -1, -1))
        ax.set_xticklabels(np.arange(8))

        # Rotate the x-axis labels
        # plt.xticks(rotation=90)
        # Add a title
        plt.title("Edit / Target Norm Heatmap")

        target_attn_mask = batch["target_attention_mask"][batch_index]
        if stopping_index is None:
            editing_target_tokens = batch["target_input_ids"][batch_index]
        else:
            editing_target_tokens = batch["target_input_ids"][batch_index][
                target_attn_mask > 0
            ][:stopping_index]

        editing_target = tokenizer.convert_ids_to_tokens(
            editing_target_tokens,
            skip_special_tokens=True,
        )
        editor_input = tokenizer.convert_ids_to_tokens(
            batch["editor_input_ids"][batch_index], skip_special_tokens=True
        )
        select_logits = (
            result.logits[batch_index][target_attn_mask > 0][:stopping_index].cpu()
            if stopping_index
            else result.logits[batch_index][target_attn_mask > 0].cpu()
        )
        editor_preds = torch.argmax(select_logits.softmax(-1), dim=-1)
        editor_preds = tokenizer.batch_decode(editor_preds, skip_special_tokens=True)

        if show_plot:
            print("Editing target:", editing_target)
            print("Editor input:", editor_input)
            print("Editor preds:", editor_preds)

        if show_plot:
            plt.show()

        if not save_path:
            continue

        batch_path = save_path / f"batch_{batch_index}"
        os.makedirs(batch_path, exist_ok=True)

        plt.savefig(batch_path / "attn_heatmap.png")
        if not show_plot:
            plt.close()

        with open(batch_path / "preds.json", "w") as f:
            preds = {
                "editing_target": editing_target,
                "editor_input": editor_input,
                "editor_preds": editor_preds,
            }

            json.dump(preds, f)


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
