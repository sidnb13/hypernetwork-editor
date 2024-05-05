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

from helpers import concat_and_pad_ids, slice_and_move_batch_for_device
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

    edited_target_logprobs = torch.nn.functional.log_softmax(editor_out.logits, dim=-1)
    target_mask = batch["target_attention_mask"] > 0
    # compute soft labels
    with torch.no_grad():
        pad_token = editor.target_model.config.eos_token_id
        combined_input_ids = concat_and_pad_ids(batch, pad_token)
        target_logits = editor.target_model(
            input_ids=combined_input_ids, attention_mask=combined_input_ids != pad_token
        ).logits

        lengths_A = torch.sum(batch["target_input_ids"] != pad_token, dim=1)
        lengths_B = torch.sum(batch["editor_input_ids"] != pad_token, dim=1)

        # Create an empty tensor to store the predictions
        shape = (len(lengths_A), edited_target_logprobs.shape[-2], 50257)
        extracted_logits = torch.full(
            shape, torch.nan, device=edited_target_logprobs.device
        )

        # Extract the predictions corresponding to B
        for i in range(len(lengths_A)):
            extracted_logits[i, : lengths_B[i], :] = target_logits[
                i, lengths_A[i] : lengths_A[i] + lengths_B[i], :
            ]

        target_logprobs = torch.nn.functional.log_softmax(extracted_logits, dim=-1)

    # compute KL div loss
    kl_div_loss = (
        edited_target_logprobs[target_mask, :]
        * (edited_target_logprobs[target_mask, :] - target_logprobs[target_mask, :])
    ).sum(-1)

    if average:
        return kl_div_loss / target_mask.sum()

    return kl_div_loss


def compute_ce_loss(
    editor: GPT2Editor,
    batch: Dict,
    rank,
    world_size,
    stop_editing_idx: int = None,
    average_log_prob: bool = True,
    token_level: bool = False,
):
    # run the hypernetwork
    batch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = editor(**batch, stop_editing_idx=stop_editing_idx)

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

    if token_level:
        return per_token_logps * loss_mask
    elif average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
