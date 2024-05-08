import itertools
import os
from typing import Dict

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from helpers import concat_and_pad_ids, slice_and_move_batch_for_device
from logger import get_logger
from models.gpt2 import GPT2Editor
from models.utils import EditorModelOutput

logger = get_logger(__name__)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(
    rank: int,
    world_size: int,
    config: DictConfig,
    editor: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
):
    # distributed setup
    if config.train.use_ddp:
        setup(rank, world_size)
        editor = DDP(editor.to(rank), device_ids=[rank])
    else:
        editor = editor.to(rank)

    opt = torch.optim.Adam(
        editor.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
    )

    if config.train.steps > 0:
        total_steps = config.train.steps // config.train.gradient_accumulation_steps
    else:
        total_steps = (
            config.train.epochs
            * len(train_dataloader)
            // config.train.gradient_accumulation_steps
        )
    warmup_steps = (
        int(total_steps * config.train.warmup_steps)
        if (0 < config.train.warmup_steps < 1)
        else config.train.warmup_steps
    ) // config.train.gradient_accumulation_steps

    if config.train.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    elif config.train.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    elif config.train.scheduler == "warmup_constant":
        scheduler = get_constant_schedule_with_warmup(opt, warmup_steps)
    elif config.train.scheduler == "constant":
        scheduler = get_constant_schedule(opt)
    else:
        raise ValueError(f"Unknown scheduler: {config.train.scheduler}")

    # wandb setup
    if config.wandb.resume and config.wandb.run_id:
        load_model_checkpoint(rank, config.ckpt_folder, editor, opt, scheduler, config)
    elif config.wandb.enabled and rank == 0:
        wandb.init(
            project=config.wandb.project,
            name=config.exp_name,
            config=dict(config),
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            group=config.wandb.group,
        )

    # training
    train_itr = itertools.cycle(train_dataloader)
    val_itr = itertools.cycle(validation_dataloader)
    grad_acc_steps = 0
    train_examples_counter = val_examples_counter = 0

    for step in range(total_steps):
        train_batch = next(train_itr)
        # compute loss
        if config.train.loss == "kl":
            loss, kl_loss, penalty_loss = compute_kl_loss(
                editor, train_batch, rank, world_size
            )
            ce_loss = None
        else:
            loss, ce_loss, penalty_loss = compute_ce_loss(
                editor, train_batch, rank, world_size
            )
            kl_loss = None

        (loss / config.train.gradient_accumulation_steps).backward()
        grad_acc_steps += 1
        train_examples_counter += len(train_batch[next(iter(train_batch.keys()))])

        if grad_acc_steps == config.train.gradient_accumulation_steps:
            grad_acc_steps = 0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                editor.parameters(), config.train.max_grad_norm
            )
            opt.step()
            opt.zero_grad()
            scheduler.step()

            batch_metrics = {
                "loss/train": loss.detach().item()
                if loss.dim() == 0
                else loss.detach().mean().item(),
                "penalty/train": penalty_loss.detach().item(),
                "lr": opt.param_groups[0]["lr"],
                "counters/train_examples": train_examples_counter,
                "counters/step": step,
                "counters/epoch": step / len(train_dataloader),
                "grad_norm": grad_norm.item(),
            }

            if ce_loss:
                batch_metrics["ce/train"] = ce_loss.detach().item()
            if kl_loss:
                batch_metrics["kl/train"] = kl_loss.detach().item()

            if rank == 0 and (step > 0 and step % config.train.log_interval == 0):
                print(batch_metrics)
                if config.wandb.enabled:
                    wandb.log(batch_metrics)

        if step > 0 and step % config.train.eval_interval == 0:
            batch_metrics = {
                "counters/val_examples": val_examples_counter,
            }
            with torch.no_grad():
                val_batch = next(val_itr)
                if config.train.loss == "kl":
                    loss, kl_loss, penalty_loss = compute_kl_loss(
                        editor, val_batch, rank, world_size
                    )
                    batch_metrics["kl/val"] = kl_loss.detach().item()
                else:
                    loss, ce_loss, penalty_loss = compute_ce_loss(
                        editor, val_batch, rank, world_size
                    )
                    batch_metrics["ce/val"] = ce_loss.detach().item()

            batch_metrics["loss/penalty"] = penalty_loss.detach().item()
            batch_metrics["loss/val"] = (
                loss.detach().item() if loss.dim() == 0 else loss.detach().mean().item()
            )

            val_examples_counter += len(val_batch[next(iter(val_batch.keys()))])

        if (
            rank == 0
            and (step > 0 and step % config.train.save_interval == 0)
            and config.train.do_save
        ):
            save_model_checkpoint(step, editor, opt, scheduler, config)

    if config.train.use_ddp:
        cleanup()


def save_model_checkpoint(
    step,
    model: GPT2Editor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: DictConfig,
):
    """Save a model checkpoint"""
    state_dict = {
        "hypernetwork": model.hypernetwork.state_dict(),
        "step": step,
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    ckpt_folder = os.path.join(config.ckpt_dir, config.exp_name, "step-{}".format(step))
    os.makedirs(ckpt_folder, exist_ok=True)
    torch.save(state_dict, ckpt_folder)
    OmegaConf.save(config, os.path.join(ckpt_folder, "config.yaml"))


def load_model_checkpoint(
    rank: int,
    ckpt_folder: str,
    model: GPT2Editor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: DictConfig,
):
    """Load a model checkpoint and resume wandb run."""
    if rank == 0 and config.wandb.enabled and config.wandb.resume:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            resume="must",
            id=config.wandb.run_id,
        )

    state_dict = torch.load(ckpt_folder, map_location=rank)
    model.hypernetwork.load_state_dict(state_dict["hypernetwork"])
    optimizer.load_state_dict(state_dict["opt"])
    scheduler.load_state_dict(state_dict["scheduler"])


def compute_penalty_loss(out: EditorModelOutput, lam: float, edit_stop_idx: int):
    if edit_stop_idx is not None:
        edit_vector_norm = out.edit_vectors.norm(dim=-1)[:, :edit_stop_idx]
    else:
        edit_vector_norm = out.edit_vectors.norm(dim=-1)
    edit_ratio = edit_vector_norm / out.target_hidden_states.norm(dim=-1)
    per_datapoint_penalty_loss = lam * torch.sum(edit_ratio, dim=[1, 2])

    return per_datapoint_penalty_loss


def compute_kl_loss(
    editor,
    batch: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    stop_editing_idx: int = None,
    lam: float = 0.0,
):
    # run the hypernetwork
    batch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = editor(
        **batch,
        stop_editing_idx=stop_editing_idx,
        output_target_hidden_states=True,
        output_edited_hidden_states=True,
        output_edit_vectors=True,
    )

    edited_target_logps = torch.nn.functional.log_softmax(editor_out.logits, dim=-1)
    edit_target_mask = batch["target_attention_mask"] > 0

    # compute soft labels
    with torch.no_grad():
        target_model = (
            editor.module.target_model
            if isinstance(editor, DDP)
            else editor.target_model
        )

        tok = AutoTokenizer.from_pretrained(target_model.config.name_or_path)

        pad_token = target_model.config.eos_token_id
        combined_input_ids = concat_and_pad_ids(batch, pad_token)

        print("BATCH TARGET IDS", batch["target_input_ids"])
        print("BATCH EDITED IDS", batch["editor_input_ids"])
        print("COMBINED INPUT IDS", combined_input_ids)

        # tokenize them
        print(
            "TARGET DECODE",
            tok.batch_decode(batch["target_input_ids"], skip_special_tokens=True),
        )
        print(
            "EDITOR DECODE",
            tok.batch_decode(batch["editor_input_ids"], skip_special_tokens=True),
        )
        print(
            "COMBINED DECODE",
            tok.batch_decode(combined_input_ids, skip_special_tokens=True),
        )

        target_logits = target_model(
            input_ids=combined_input_ids, attention_mask=combined_input_ids != pad_token
        ).logits

        lengths_A = torch.sum(batch["editor_attention_mask"] > 0, dim=1)
        lengths_B = torch.sum(batch["target_attention_mask"] > 0, dim=1)

        # Create an empty tensor to store the predictions
        target_seq_len = edited_target_logps.shape[-2]
        edit_seq_len = target_logits.shape[-2]
        shape = (
            len(lengths_A),
            target_seq_len,
            target_model.config.vocab_size,
        )
        extracted_logits = torch.full(
            shape, torch.nan, device=edited_target_logps.device
        )
        # Extract the predictions corresponding to B, assume LEFT padding
        for i in range(len(lengths_A)):
            extracted_logits[i, target_seq_len - lengths_B[i] :, :] = target_logits[
                i, edit_seq_len - lengths_B[i] :, :
            ]

        target_logps = torch.nn.functional.log_softmax(extracted_logits, dim=-1)

    print("EXTRACT LOGITS SHAPE", extracted_logits.shape)

    print("TARGET INPUT IDS", batch["target_input_ids"][edit_target_mask])
    print("COMBINED INPUT IDS", combined_input_ids[:, -50:][edit_target_mask])
    print("EDIT TARGET MASK", edit_target_mask)

    edited_target_preds = torch.argmax(editor_out.logits[edit_target_mask, :], dim=-1)
    target_preds = torch.argmax(extracted_logits[edit_target_mask, :], dim=-1)

    print("EDITED TARGET PREDICTIONS", "".join(tok.batch_decode(edited_target_preds)))
    print("TARGET PREDICTIONS", "".join(tok.batch_decode(target_preds)))

    # compute KL div loss
    kl_div_loss = (
        edited_target_logps[edit_target_mask, :].exp()
        * (edited_target_logps[edit_target_mask, :] - target_logps[edit_target_mask, :])
    ).mean()
    
    

    print("KL:", kl_div_loss.mean().item())

    exit(0)

    penalty_loss = compute_penalty_loss(editor_out, lam, stop_editing_idx)

    # gather from all ranks
    if dist.is_initialized():
        dist.all_reduce(kl_div_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(penalty_loss, op=dist.ReduceOp.SUM)

    return (
        kl_div_loss.mean() + penalty_loss.mean(),
        kl_div_loss.mean(),
        penalty_loss.mean(),
    )


def compute_ce_loss(
    editor,
    batch: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    stop_editing_idx: int = None,
    average_log_prob: bool = True,
    lam: float = 0.0,
):
    # run the hypernetwork
    batch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = editor(
        **batch,
        stop_editing_idx=stop_editing_idx,
        output_target_hidden_states=True,
        output_edited_hidden_states=True,
        output_edit_vectors=True,
    )

    loss_mask = batch["target_attention_mask"] > 0
    labels = torch.where(loss_mask, batch["target_input_ids"], 0)
    labels = labels[:, 1:].clone()
    loss_mask = loss_mask[:, 1:].clone()
    logits = editor_out.logits[:, :-1, :]

    # compute ce loss
    distribution_logps = logits.log_softmax(-1)
    per_token_logps = torch.gather(
        distribution_logps, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze()

    if average_log_prob:
        ce_loss = -(per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        ce_loss = -(per_token_logps * loss_mask).sum(-1)

    penalty_loss = compute_penalty_loss(editor_out, lam, stop_editing_idx)

    # gather from all ranks
    if dist.is_initialized():
        dist.all_reduce(ce_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(penalty_loss, op=dist.ReduceOp.SUM)

    loss = ce_loss.mean() + penalty_loss.mean()

    return loss, ce_loss.mean(), penalty_loss.mean()
