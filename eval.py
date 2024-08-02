"""Evaluation harness to test editor models."""

import os
from collections import defaultdict
from contextlib import nullcontext

import attr
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer

from helpers import slice_and_move_batch_for_device, visualize_interventions
from logger import get_logger
from models.base import BaseEditor
from models.utils import add_fwd_hooks

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(config, model: BaseEditor, dataloader):
    model = model.to(torch.cuda.current_device())
    summary_metrics = defaultdict(list)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    for eval_step, batch in enumerate(tqdm(dataloader)):
        batch = slice_and_move_batch_for_device(batch, 0, 1)
        if config.eval.enable_editor:
            editor_output = model(
                editor_input_ids=batch["editor_input_ids"],
                editor_attention_mask=batch["editor_attention_mask"],
                target_input_ids=batch["target_input_ids"],
                target_attention_mask=batch["target_attention_mask"],
                stop_editing_idx=config.train.stop_editing_idx,
                output_hooks=True,
                output_target_hidden_states=True,
                output_edit_vectors=True,
                output_editor_attention=True,
            )
        else:
            editor_output = None
        if editor_output and eval_step % config.eval.visualize_interval == 0:
            target_out = model.target_model(
                input_ids=batch["target_input_ids"],
                attention_mask=batch["target_attention_mask"],
            )
            visualize_interventions(
                result=editor_output,
                orig_logits=target_out.logits,
                batch=batch,
                save_path=os.path.join(
                    config.eval_dir, config.exp_name, "step-{}".format(eval_step)
                ),
                tokenizer=tokenizer,
                stopping_index=config.train.stop_editing_idx,
                metadata=config,
            )
        with add_fwd_hooks(editor_output.hooks) if editor_output else nullcontext():
            if not config.eval.enable_editor:
                # concatenate editor input and target input, left pad at the end
                target_input_ids = concat_and_left_pad(
                    batch["editor_input_ids"],
                    batch["editor_attention_mask"],
                    batch["target_input_ids"],
                    batch["target_attention_mask"],
                    pad_value=tokenizer.pad_token_id,
                )
                target_attention_mask = concat_and_left_pad(
                    batch["editor_attention_mask"],
                    batch["editor_attention_mask"],
                    batch["target_attention_mask"],
                    batch["target_attention_mask"],
                    pad_value=0,
                )
            else:
                target_input_ids = batch["target_input_ids"]
                target_attention_mask = batch["target_attention_mask"]
            generation_results = model.target_model.generate(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                max_new_tokens=config.eval.max_new_tokens,
                do_sample=True,
                top_p=config.eval.top_p,
                top_k=config.eval.top_k,
                temperature=config.eval.temperature,
            )
            input_shape = batch["target_input_ids"].shape[1]
            generation_results = generation_results[:, input_shape:]

        # decode
        decoded_generation_results = tokenizer.batch_decode(
            generation_results, skip_special_tokens=True
        )
        decoded_targets = tokenizer.batch_decode(
            batch["target_input_ids"], skip_special_tokens=True
        )

        batch_metrics = np.array(
            [
                [
                    compute_exact_match(gen, tgt),
                    compute_recall(gen, tgt),
                    compute_f1(gen, tgt),
                ]
                for gen, tgt in zip(decoded_generation_results, decoded_targets)
            ]
        )
        summary_metrics["exact_match"].extend(batch_metrics[:, 0].flatten().tolist())
        summary_metrics["recall"].extend(batch_metrics[:, 1].flatten().tolist())
        summary_metrics["f1"].extend(batch_metrics[:, 2].flatten().tolist())

    for k, v in summary_metrics.items():
        summary_metrics[k] = np.mean(v)

    logger.info(
        f"Exact match: {summary_metrics['exact_match']:.4f}, "
        f"Recall: {summary_metrics['recall']:.4f}, "
        f"F1: {summary_metrics['f1']:.4f}"
    )


def concat_and_left_pad(tensor1, mask1, tensor2, mask2, pad_value=0):
    lengths1 = mask1.sum(dim=1)
    lengths2 = mask2.sum(dim=1)
    concatenated = [
        torch.cat([t1[:l1], t2[:l2]])
        for t1, l1, t2, l2 in zip(tensor1, lengths1, tensor2, lengths2)
    ]
    flipped = [t.flip(0) for t in concatenated]
    padded = pad_sequence(flipped, batch_first=True, padding_value=pad_value)
    result = padded.flip(1)
    return result


def compute_f1(prediction_tokens, ground_truth_tokens):
    """
    Compute F1 score between prediction and ground truth tokens.
    """
    prediction_set = set(prediction_tokens)
    ground_truth_set = set(ground_truth_tokens)

    intersection = prediction_set.intersection(ground_truth_set)
    precision_score = len(intersection) / len(prediction_set)
    recall_score = len(intersection) / len(ground_truth_set)

    if precision_score + recall_score == 0:
        return 0.0

    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    return f1_score


def compute_recall(prediction_tokens, ground_truth_tokens):
    """
    Compute recall score between prediction and ground truth tokens.
    """
    prediction_set = set(prediction_tokens)
    ground_truth_set = set(ground_truth_tokens)
    intersection = prediction_set.intersection(ground_truth_set)
    recall_score = len(intersection) / len(ground_truth_set)
    return recall_score


def compute_exact_match(prediction_tokens, ground_truth_tokens):
    """
    Compute exact match score between prediction and ground truth tokens.
    """
    if prediction_tokens == ground_truth_tokens:
        return 1.0
    else:
        return 0.0
