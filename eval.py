"""Evaluation harness to test editor models."""

import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from helpers import slice_and_move_batch_for_device, visualize_interventions
from models.base import BaseEditor
from models.utils import add_fwd_hooks


@torch.no_grad()
def evaluate(config, model: BaseEditor, dataloader):
    model = model.to(torch.cuda.current_device())

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    for eval_step, batch in enumerate(tqdm(dataloader)):
        batch = slice_and_move_batch_for_device(batch, 0, 1)
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
        if eval_step % config.eval.visualize_interval == 0:
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
        with add_fwd_hooks(editor_output.hooks):
            generation_results = model.target_model.generate(
                input_ids=batch["target_input_ids"],
                attention_mask=batch["target_attention_mask"],
                max_new_tokens=config.eval.max_new_tokens,
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

        breakpoint()


def compute_f1(preds, targets):
    pass


def compute_em(preds, targets):
    pass
