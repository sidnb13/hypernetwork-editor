import itertools
from typing import Dict, Union

import torch
import torch.distributed as dist
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
from models.gpt2 import GPT2Editor
from models.utils import EditorModelOutput


def train(
    config: DictConfig,
    editor: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
):
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

    for step in range(total_steps):
        train_batch = next(train_itr)
        # forward pass
        target_out = editor(**slice_and_move_batch_for_device(train_batch))
        # compute loss
        if config.train.loss == "kl":
            loss = compute_kl_loss(editor, train_batch, target_out)
        else:
            loss = compute_ce_loss(train_batch, target_out)


def compute_kl_loss(editor: GPT2Editor, batch: Dict, out: EditorModelOutput):
    loss_mask = batch["editor_attention_mask"] > 0
    editor_logprobs = torch.nn.functional.log_softmax(out.logits)
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
    kl_div_loss = (editor_logprobs * (editor_logprobs - target_logprobs)).sum(
        -1
    ) / loss_mask.sum(-1)

    return kl_div_loss


def compute_ce_loss(editor: nn.Module, batch: Dict, out: EditorModelOutput):
    pass
