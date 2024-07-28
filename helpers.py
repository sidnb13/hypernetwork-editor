import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from models.utils import EditorModelOutput

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM2WORD = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}
COLOR_MAP = {
    "g": "green",
    "b": "brown",
    "r": "red",
    "y": "yellow",
    "p": "purple",
    "o": "orange",
}


def compute_l0_l1_norms(tensor):
    # Compute L0 norm (number of non-zero elements)
    l0_norm = torch.count_nonzero(tensor).item() / tensor.numel()
    # Compute L1 norm (sum of absolute values)
    l1_norm = torch.norm(tensor, p=1).item()
    return l0_norm, l1_norm


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # bind to all interfaces and use an OS provided port
        return s.getsockname()[1]  # return only the port number


def compute_stop_mask(target_attention_mask, stop_editing_idx):
    first_true = target_attention_mask.argmax(-1, keepdim=True)
    shifts = (target_attention_mask.shape[-1] - first_true) - stop_editing_idx
    indices = (
        first_true
        + shifts
        + torch.arange(stop_editing_idx, device=target_attention_mask.device).unsqueeze(
            0
        )
    )

    if target_attention_mask.dim() == 1:
        indices = indices.squeeze()

    target_attention_mask = target_attention_mask.clone().scatter_(
        -1,
        indices,
        True,
    )
    csum = target_attention_mask.cumsum(-1)
    stop_edit_mask = torch.where(
        csum < stop_editing_idx,
        target_attention_mask,
        torch.logical_and(csum > 0, csum <= stop_editing_idx),
    ).bool()

    return stop_edit_mask


def visualize_interventions(
    result: EditorModelOutput,
    orig_logits: torch.Tensor,
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
        # The tensor norm comes in an stopping_index * num_layers+1 matrix
        target_attn_mask = batch["target_attention_mask"][batch_index]
        target_input_ids = batch["target_input_ids"][batch_index]
        result_logits = result.logits[batch_index]

        using_ghost_token = False
        # Add in the ghost token to the mask and target id's
        if target_attn_mask.shape[0] == result.edit_vectors[batch_index].shape[0] - 1:
            using_ghost_token = True
            target_attn_mask = torch.cat(
                [torch.ones(1, device=target_attn_mask.device), target_attn_mask]
            )
            # also add a placeholder token id = 0 to the input_ids
            target_input_ids = torch.cat(
                [
                    torch.zeros(
                        1, device=target_input_ids.device, dtype=target_input_ids.dtype
                    ),
                    target_input_ids,
                ]
            )

        target_attn_mask = compute_stop_mask(target_attn_mask, stopping_index)
        edit_tensor = result.edit_vectors[batch_index][target_attn_mask > 0].cpu()
        target_hidden = result.target_hidden_states[batch_index].cpu()

        edit_tensor[:stopping_index, :, :] = edit_tensor[
            :stopping_index, :, :
        ] / target_hidden[:stopping_index].norm(dim=2, keepdim=True)
        edit_tensor_norm = edit_tensor.norm(dim=2).flip(1)

        # is this any better??
        # attention_matrix = result['editor_attention'][batch_index].reshape(104).to("cpu").reshape(13,8).permute(1,0)

        # Detach and convert to numpy
        edit_tensor_norm = edit_tensor_norm.numpy()[:stopping_index, :]

        editing_target_tokens = target_input_ids[target_attn_mask > 0]
        if stopping_index is not None:
            editing_target_tokens = editing_target_tokens[:stopping_index]

        editing_target = tokenizer.batch_decode(
            editing_target_tokens,
            skip_special_tokens=True,
        )
        editor_input = tokenizer.batch_decode(
            batch["editor_input_ids"][batch_index][
                batch["editor_attention_mask"][batch_index] > 0
            ],
            skip_special_tokens=True,
        )
        if using_ghost_token:
            selection = target_attn_mask > 0
            select_logits = (
                result_logits[selection[1:]][:stopping_index].cpu()
                if stopping_index
                else result_logits[target_attn_mask > 0].cpu()
            )
        else:
            select_logits = (
                result_logits[target_attn_mask > 0][:stopping_index].cpu()
                if stopping_index
                else result_logits[target_attn_mask > 0].cpu()
            )
        editor_preds = torch.argmax(select_logits.softmax(-1), dim=-1)
        editor_preds = tokenizer.batch_decode(editor_preds, skip_special_tokens=True)

        # model without intervention
        orig_preds = torch.argmax(orig_logits[batch_index].softmax(-1), dim=-1)
        orig_preds = tokenizer.batch_decode(orig_preds, skip_special_tokens=True)

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap = ax.imshow(edit_tensor_norm.transpose(), cmap="hot")

        # Color the heatmap according to the entry sizes
        heatmap.set_clim(vmin=np.min(0), vmax=np.max(edit_tensor_norm))
        cbar = plt.colorbar(heatmap)
        cbar.set_label("Entry Sizes")

        # Decode tokens for x-axis labels
        x_axis_tokens = tokenizer.batch_decode(
            editing_target_tokens, skip_special_tokens=True
        )

        # Add labels to the x and y axes
        ax.set_yticks(np.arange(13))
        ax.set_xticks(np.arange(len(x_axis_tokens)))
        ax.set_yticklabels(np.arange(12, -1, -1))
        ax.set_xticklabels(x_axis_tokens, rotation=45, ha="right")

        # Add a title
        plt.title("Edit / Target Norm Heatmap")

        # Adjust layout
        plt.tight_layout()

        if show_plot:
            print("Editing target:", editing_target)
            print("Editor input:", editor_input)
            print("Editor preds:", editor_preds)
            print("Orig preds:", orig_preds)

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
                "orig_preds": orig_preds,
            }

            json.dump(preds, f)


def get_nb_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_bytes = (
                param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            )
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_tokenizer(name_or_path, padding_side="right"):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token_id = tok.eos_token_id
    # This is very important because we take last hidden state in editor
    tok.padding_side = padding_side
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
    batch_size, _ = first.size()
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
    # Concatenate A[i] and B[i] a, assume RIGHT padding
    for i in range(batch_size):
        result[i, : lengths_A[i]] = first[i, : lengths_A[i]]
        result[i, lengths_A[i] : lengths_A[i] + lengths_B[i] :] = second[
            i, : lengths_B[i]
        ]

    return result
