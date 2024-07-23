import itertools
import os
from typing import Dict

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from helpers import cleanup, setup, slice_and_move_batch_for_device
from logger import get_logger
from models.gpt2 import GPT2Editor

logger = get_logger(__name__)


def finetune(
    rank: int,
    world_size: int,
    config: DictConfig,
    editor: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader = None,
):
    # distributed setup
    if config.train.use_ddp:
        setup(rank, world_size)
        editor = DDP(editor.to(rank), device_ids=[rank])
    else:
        editor = editor.to(rank)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    opt = torch.optim.AdamW(
        # editor.parameters(),
        [
            param for param in editor.parameters() if param.requires_grad
        ],  # this still is not sufficient to shut things off though...
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
    )

    if config.train.steps > 0:
        total_steps = config.train.steps // config.train.gradient_accumulation_steps
    else:
        total_steps = (
            int(config.train.n_epochs * len(train_dataloader))
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

    start_step = 0

    # wandb setup
    if (config.wandb.resume and config.wandb.run_id) or config.resume_ckpt:
        logger.info("Resuming run from checkpoint")
        start_step = load_model_checkpoint(
            rank, config.resume_ckpt, editor, opt, scheduler, config
        )
    elif config.wandb.enabled and rank == 0 and not config.debug:
        wandb.init(
            project=config.wandb.project,
            name=config.exp_name,
            notes=config.wandb.notes,
            config=OmegaConf.to_container(config),
            entity=config.wandb.entity,
            tags=config.wandb.tags,
            group=config.wandb.group,
        )

    # training
    train_itr = itertools.cycle(train_dataloader)
    if validation_dataloader is not None:
        val_itr = itertools.cycle(validation_dataloader)

    # skip start_steps
    for i in range(start_step):
        next(train_itr)
        if val_itr and i > 0 and i % config.train.eval_interval == 0:
            next(val_itr)

    if start_step > 0:
        logger.info(f"Skipped {start_step} steps...")

    grad_acc_steps = 0
    updates = 0
    train_examples_counter = val_examples_counter = 0

    logger.info(
        f"Training for {total_steps - start_step} ({start_step}, {total_steps}) steps..."
    )

    for step in range(start_step, total_steps):
        # set model to training mode
        if isinstance(editor, DDP):
            editor.module.train()
        else:
            editor.train()

        train_batch = next(train_itr)

        # compute loss
        loss = compute_ce_loss(
            editor,
            train_batch,
            rank,
            world_size,
        )

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
            updates += 1

            loss_reduced = loss.detach() if loss.dim() == 0 else loss.detach().mean()

            batch_metrics = {
                "loss/train": loss_reduced.item(),
                "loss/ppl": loss_reduced.exp().item(),
                "lr": opt.param_groups[0]["lr"],
                "counters/train_examples": train_examples_counter,
                "counters/step": step,
                "counters/updates": updates,
                "counters/epoch": step / len(train_dataloader),
                "grad_norm": grad_norm.item(),
            }

            if rank == 0 and (step > 0 and step % config.train.log_interval == 0):
                logger.info(batch_metrics)
                if wandb.run:
                    wandb.log(batch_metrics)

        if config.train.do_eval and step > 0 and step % config.train.eval_interval == 0:
            if val_itr:
                evaluate(
                    step,
                    val_examples_counter,
                    editor,
                    tokenizer,
                    val_itr,
                    rank,
                    world_size,
                    config,
                )

        if config.train.do_save and step > 0 and step % config.train.save_interval == 0:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                save_model_checkpoint(step, editor, opt, scheduler, config)

    logger.info("Finished training")

    if wandb.run and rank == 0:
        wandb.finish()

    if config.train.do_save:
        logger.info("Saving final model checkpoint")
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            save_model_checkpoint(step + 1, editor, opt, scheduler, config)

    if config.train.use_ddp:
        cleanup()


@torch.no_grad()
def evaluate(
    step,
    val_examples_counter,
    editor: torch.nn.Module,
    tokenizer,
    val_itr,
    rank,
    world_size,
    config,
):
    # set model to training mode
    if isinstance(editor, DDP):
        editor.module.eval()
    else:
        editor.eval()
    batch_metrics = {
        "counters/val_examples": val_examples_counter,
        "counters/step": step,
    }
    val_batch = next(val_itr)
    loss = compute_ce_loss(
        editor,
        val_batch,
        rank,
        world_size,
    )

    loss_reduced = loss.detach() if loss.dim() == 0 else loss.detach().mean()

    batch_metrics["val/ppl"] = loss_reduced.exp().item()
    batch_metrics["loss/val"] = loss_reduced.item()
    logger.info(batch_metrics)

    if wandb.run and rank == 0:
        wandb.log(batch_metrics)


def save_model_checkpoint(
    step,
    model: PreTrainedModel | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: DictConfig,
):
    """Save a model checkpoint"""
    if config.debug:
        return

    model_obj = model.module if isinstance(model, DDP) else model
    state_dict = {
        "state": model_obj.state_dict(),
        "step": step,
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    ckpt_folder = os.path.join(config.ckpt_dir, config.exp_name, "step-{}".format(step))
    os.makedirs(ckpt_folder, exist_ok=True)
    torch.save(state_dict, os.path.join(ckpt_folder, "checkpoint.pt"))

    logger.info("Saved model checkpoint to {}".format(ckpt_folder))

    OmegaConf.save(config, os.path.join(ckpt_folder, "config.yaml"))


def load_model_checkpoint(
    rank: int,
    ckpt_folder: str,
    model: GPT2Editor | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: DictConfig,
) -> int:
    """Load a model checkpoint and resume wandb run."""
    if rank == 0 and config.wandb.enabled and config.wandb.run_id:
        logger.info("Resuming wandb run")
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            resume="must",
            id=config.wandb.run_id,
        )

    state_dict = torch.load(
        os.path.join(ckpt_folder, "checkpoint.pt"), map_location=torch.device(rank)
    )
    model_obj = model.module if isinstance(model, DDP) else model
    model_obj.load_state_dict(state_dict["state"])
    logger.info("Loaded model checkpoint from {}".format(ckpt_folder))
    optimizer.load_state_dict(state_dict["opt"])
    scheduler.load_state_dict(state_dict["scheduler"])

    return state_dict["step"]


def compute_ce_loss(
    model,
    batch: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    average_log_prob: bool = True,
):
    # run the hypernetwork
    batch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = model(**batch)

    loss_mask = batch["attention_mask"] > 0
    labels = torch.where(loss_mask, batch["input_ids"], 0)
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
    ce_loss = ce_loss.mean()

    # gather from all ranks
    if dist.is_initialized():
        dist.all_reduce(ce_loss, op=dist.ReduceOp.SUM)

    ce_loss = ce_loss / world_size

    return ce_loss
