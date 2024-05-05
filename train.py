import itertools
from typing import Dict

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from helpers import slice_and_move_batch_for_device
from logger import get_logger
from models.gpt2 import GPT2Editor

logger = get_logger(__name__)


def train(
    config: DictConfig,
    editor: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    rank: int,
    world_size: int,
):
    # wandb setup
    if config.wandb.enabled and rank == 0:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            config=dict(config),
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            group=config.wandb.group,
        )

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

    # training
    train_itr = itertools.cycle(train_dataloader)
    val_itr = itertools.cycle(validation_dataloader)
    grad_acc_steps = 0
    examples_counter = 0

    for step in range(total_steps):
        train_batch = next(train_itr)
        # compute loss
        if config.train.loss == "kl":
            loss = compute_kl_loss(editor, train_batch, rank, world_size)
        else:
            loss = compute_ce_loss(editor, train_batch, rank, world_size)

        (loss / config.train.gradient_accumulation_steps).backward()
        grad_acc_steps += 1
        examples_counter += len(train_batch[next(iter(train_batch.keys()))])

        batch_metrics = {
            "loss/train": loss.detach().item()
            if loss.shape[0] == 1
            else loss.detach().mean().item(),
            "lr": opt.param_groups[0]["lr"],
            "counters/examples": examples_counter,
            "counters/step": step,
            "counters/epoch": step / len(train_dataloader),
        }

        if grad_acc_steps == config.train.gradient_accumulation_steps == 0:
            grad_acc_steps = 0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                editor.parameters(), config.train.max_grad_norm
            )
            batch_metrics["grad_norm"] = grad_norm.item()
            opt.step()
            opt.zero_grad()
            scheduler.step()

        if (
            config.train.log_interval > 0
            and step % config.train.log_interval == 0
            and rank == 0
        ):
            logger.info(batch_metrics)
            if config.wandb.enabled:
                wandb.log(batch_metrics)


def compute_kl_loss(
    editor: GPT2Editor,
    batch: Dict,
    rank,
    world_size,
    average: bool = True,
):
    # run the hypernetwork
    batch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = editor(**batch)

    loss_mask = batch["editor_attention_mask"] > 0
    editor_logprobs = torch.nn.functional.log_softmax(editor_out.logits)
    # compute soft labels
    with torch.no_grad():
        # concat and pad outputs
        combined_inputs = torch.full_like(loss_mask, -1)
        nonpad_target = batch["editor_attention_mask"] > 0
        combined_inputs[nonpad_target] = batch["target_input_ids"][nonpad_target]
        combined_inputs[loss_mask] = batch["editor_attention_mask"][loss_mask]
        # run target model
        target_logits = editor.target_model(
            input_ids=combined_inputs,
            attention_mask=torch.where(combined_inputs > 0, 1, 0).to(loss_mask.device),
        ).logits
        target_logprobs = torch.nn.functional.log_softmax(target_logits)

    # compute KL div loss
    kl_div_loss = (editor_logprobs * (editor_logprobs - target_logprobs)).sum(-1)

    if average:
        return kl_div_loss / loss_mask.sum(-1)

    return kl_div_loss


def compute_ce_loss(
    editor: GPT2Editor,
    batch: Dict,
    rank,
    world_size,
    edit_stop_idx: int = None,
    average_log_prob: bool = True,
    token_level: bool = False,
):
    # run the hypernetwork
    batch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = editor(**batch, edit_stop_idx=edit_stop_idx)

    loss_mask = batch["target_attention_mask"] > 0
    labels = torch.where(loss_mask, batch["target_input_ids"], 0)
    labels = labels[:, 1:].clone()
    logits = editor_out.logits[:, :-1, :]

    # compute ce loss
    distribution_logps = logits.log_softmax(-1)
    per_token_logps = torch.gather(
        distribution_logps, dim=-1, index=labels.unsqueeze(-1)
    )

    if token_level:
        return per_token_logps * loss_mask
    elif average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
