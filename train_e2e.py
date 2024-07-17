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
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from helpers import (
    cleanup,
    compute_l0_l1_norms,
    concat_and_pad_ids,
    setup,
    slice_and_move_batch_for_device,
    visualize_interventions,
)
from logger import get_logger
from models.gpt2 import GPT2Editor
from models.utils import EditorModelOutput

logger = get_logger(__name__)


def train(
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
    else:
        val_itr = None

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
        if config.train.loss == "kl":
            loss, kl_loss, penalty_loss, out = compute_kl_loss(
                editor,
                train_batch,
                rank,
                world_size,
                stop_editing_idx=config.train.stop_editing_idx,
            )
            ce_loss = None
        else:
            loss, ce_loss, penalty_loss, out = compute_ce_loss(
                editor,
                train_batch,
                rank,
                world_size,
                stop_editing_idx=config.train.stop_editing_idx,
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
            # #### Don't forget to delete this!! this zero's out the cross-attention learning entirely!!
            # for name, param in editor.named_parameters():
            #     if 'transformer.h' in name and 'crossattention' in name:
            #         param.data.zero_()  # Zero out the data
            #         param.requires_grad = False  # Disable gradients for these parameters
            #         if param.grad is not None:
            #             param.grad.zero_()  # Zero out the gradients if they exist
            opt.step()
            opt.zero_grad()
            scheduler.step()
            updates += 1

            l0, l1 = compute_l0_l1_norms(out.edit_vectors)
            ppl = compute_perplexity(train_batch, out)

            batch_metrics = {
                "loss/train": loss.detach().item()
                if loss.dim() == 0
                else loss.detach().mean().item(),
                "loss/ppl": ppl,
                "penalty/train": penalty_loss.detach().item(),
                "train_metrics/l1": l1,
                "train_metrics/l0": l0,
                "lr": opt.param_groups[0]["lr"],
                "counters/train_examples": train_examples_counter,
                "counters/step": step,
                "counters/updates": updates,
                "counters/epoch": step / len(train_dataloader),
                "grad_norm": grad_norm.item(),
            }

            if ce_loss:
                batch_metrics["ce/train"] = ce_loss.detach().item()
            if kl_loss:
                batch_metrics["kl/train"] = kl_loss.detach().item()

            if rank == 0 and (step > 0 and step % config.train.log_interval == 0):
                logger.info(batch_metrics)
                if wandb.run:
                    wandb.log(batch_metrics)

        if step > 0 and step % config.train.eval_interval == 0:
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
            else:
                visualize(
                    editor,
                    tokenizer,
                    train_batch,
                    step,
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
    if config.train.loss == "kl":
        loss, kl_loss, penalty_loss, out = compute_kl_loss(
            editor,
            val_batch,
            rank,
            world_size,
            stop_editing_idx=config.train.stop_editing_idx,
        )
        batch_metrics["kl/val"] = kl_loss.detach().item()
    else:
        loss, ce_loss, penalty_loss, out = compute_ce_loss(
            editor,
            val_batch,
            rank,
            world_size,
            stop_editing_idx=config.train.stop_editing_idx,
        )
        batch_metrics["ce/val"] = ce_loss.detach().item()

    l0, l1 = compute_l0_l1_norms(out.edit_vectors)

    batch_metrics["val/ppl"] = compute_perplexity(val_batch, out)
    batch_metrics["val_metrics/l1"] = l1
    batch_metrics["val_metrics/l0"] = l0
    batch_metrics["loss/penalty"] = penalty_loss.detach().item()
    batch_metrics["loss/val"] = (
        loss.detach().item() if loss.dim() == 0 else loss.detach().mean().item()
    )
    logger.info(batch_metrics)

    if wandb.run and rank == 0:
        wandb.log(batch_metrics)

    if not config.debug:
        visualize(
            editor,
            tokenizer,
            val_batch,
            step,
            rank,
            world_size,
            config,
        )


@torch.no_grad()
def visualize(
    editor: GPT2Editor,
    tokenizer,
    batch: Dict[str, torch.Tensor],
    step,
    rank,
    world_size,
    config,
):
    # save intervention visualizations
    local_minibatch = slice_and_move_batch_for_device(batch, rank, world_size)
    editor_out = editor(
        **local_minibatch,
        stop_editing_idx=config.train.stop_editing_idx,
        output_target_hidden_states=True,
        output_edit_vectors=True,
        output_editor_attention=True,
    )
    base = (
        editor.module.target_model if isinstance(editor, DDP) else editor.target_model
    )

    orig_out = base(
        input_ids=local_minibatch["target_input_ids"],
        attention_mask=local_minibatch["target_attention_mask"],
    )
    logger.debug(f"Saving attention heatmaps for step {step}")
    # gather results from ranks, use global batch for predictions
    if dist.is_initialized():
        if rank == 0:
            gathered_logits = [
                torch.zeros_like(editor_out.logits) for _ in range(world_size)
            ]
            gathered_target_hidden_states = [
                torch.zeros_like(editor_out.target_hidden_states)
                for _ in range(world_size)
            ]
            gathered_edit_vectors = [
                torch.zeros_like(editor_out.edit_vectors) for _ in range(world_size)
            ]
            gathered_editor_attention = [
                torch.zeros_like(editor_out.editor_attention) for _ in range(world_size)
            ]
            gathered_orig_logits = [
                torch.zeros_like(orig_out.logits) for _ in range(world_size)
            ]
        else:
            (
                gathered_logits,
                gathered_target_hidden_states,
                gathered_edit_vectors,
                gathered_editor_attention,
                gathered_orig_logits,
            ) = None, None, None, None, None

        dist.gather(editor_out.logits, gathered_logits)
        dist.gather(editor_out.target_hidden_states, gathered_target_hidden_states)
        dist.gather(editor_out.edit_vectors, gathered_edit_vectors)
        dist.gather(editor_out.editor_attention, gathered_editor_attention)
        dist.gather(orig_out.logits, gathered_orig_logits)

    if rank == 0:
        if dist.is_initialized():
            editor_out.logits = torch.cat(gathered_logits, dim=0)
            editor_out.target_hidden_states = torch.cat(
                gathered_target_hidden_states, dim=0
            )
            editor_out.edit_vectors = torch.cat(gathered_edit_vectors, dim=0)
            editor_out.editor_attention = torch.cat(gathered_editor_attention, dim=0)
            orig_out.logits = torch.cat(gathered_orig_logits, dim=0)

        visualize_interventions(
            result=editor_out,
            orig_logits=orig_out.logits,
            batch=batch,
            save_path=os.path.join(
                config.ckpt_dir, config.exp_name, "step-{}".format(step)
            )
            if config.train.do_save
            else None,
            tokenizer=tokenizer,
            stopping_index=config.train.stop_editing_idx,
            metadata=config,
        )


def save_model_checkpoint(
    step,
    model: GPT2Editor | DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: DictConfig,
):
    """Save a model checkpoint"""
    if config.debug:
        return

    model_obj = model.module if isinstance(model, DDP) else model
    state_dict = {
        "hypernetwork": model_obj.hypernetwork.state_dict(),
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
    model_obj.load_state_dict(state_dict["hypernetwork"])
    logger.info("Loaded model checkpoint from {}".format(ckpt_folder))
    optimizer.load_state_dict(state_dict["opt"])
    scheduler.load_state_dict(state_dict["scheduler"])

    return state_dict["step"]


def compute_penalty_loss(out: EditorModelOutput, lam: float, edit_stop_idx: int = None):
    if edit_stop_idx is not None:
        edit_vector_norm = out.edit_vectors.norm(dim=-1)[:, :edit_stop_idx]
    else:
        edit_vector_norm = out.edit_vectors.norm(dim=-1)
    edit_ratio = edit_vector_norm / out.target_hidden_states.norm(dim=-1)
    per_datapoint_penalty_loss = lam * torch.sum(edit_ratio, dim=[1, 2])

    return per_datapoint_penalty_loss


def compute_perplexity(batch: Dict, editor_out: EditorModelOutput):
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

    ce_loss = -(per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    ce_loss = ce_loss.mean()

    return ce_loss.exp()


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

        pad_token = target_model.config.eos_token_id
        combined_input_ids = concat_and_pad_ids(batch, pad_token)

        target_logits = target_model(
            input_ids=combined_input_ids, attention_mask=combined_input_ids != pad_token
        ).logits

        lengths_A = torch.sum(batch["editor_attention_mask"] > 0, dim=1)
        lengths_B = torch.sum(batch["target_attention_mask"] > 0, dim=1)

        # Create an empty tensor to store the predictions
        target_seq_len = edited_target_logps.shape[-2]
        shape = (
            len(lengths_A),
            target_seq_len,
            target_model.config.vocab_size,
        )
        extracted_logits = torch.full(
            shape, torch.nan, device=edited_target_logps.device
        )
        # Extract the predictions corresponding to B
        for i in range(len(lengths_A)):
            extracted_logits[i, : lengths_B[i], :] = target_logits[
                i, lengths_A[i] : lengths_A[i] + lengths_B[i], :
            ]
        target_logps = torch.nn.functional.log_softmax(extracted_logits, dim=-1)

    # compute KL div loss
    kl_div_loss = (
        target_logps[edit_target_mask, :].exp()
        * (target_logps[edit_target_mask, :] - edited_target_logps[edit_target_mask, :])
    ).sum(-1)

    kl_div_loss = kl_div_loss.mean()
    penalty_loss = compute_penalty_loss(editor_out, lam, stop_editing_idx).mean()

    # gather from all ranks
    if dist.is_initialized():
        dist.all_reduce(kl_div_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(penalty_loss, op=dist.ReduceOp.SUM)

    # normalize kl by batch size
    kl_div_loss = kl_div_loss / world_size
    penalty_loss = penalty_loss / world_size

    return (kl_div_loss + penalty_loss, kl_div_loss, penalty_loss, editor_out)


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
    ce_loss = ce_loss.mean()

    penalty_loss = compute_penalty_loss(editor_out, lam, stop_editing_idx).mean()

    # gather from all ranks
    if dist.is_initialized():
        dist.all_reduce(ce_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(penalty_loss, op=dist.ReduceOp.SUM)

    ce_loss = ce_loss / world_size
    penalty_loss = penalty_loss / world_size

    loss = ce_loss + penalty_loss
    return loss, ce_loss, penalty_loss, editor_out
