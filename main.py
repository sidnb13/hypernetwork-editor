import os
import random
from datetime import datetime

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

import models
from data import (
    get_dataloader,
    get_task,
)
from eval import evaluate
from finetune_target import finetune
from helpers import get_nb_trainable_parameters
from logger import get_logger
from train_e2e import train

logger = get_logger(__name__)

# should I need to do this?
# trying to solve:
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.cuda.empty_cache()

    if not config.debug:
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)

    config_cls = getattr(models, config.model.config_cls)
    model_cls = getattr(models, config.model.model_cls)

    if config.mode == "train_editor" or config.mode == "eval":
        model_config = config_cls(
            _name_or_path=config.model.name_or_path,
            edit_channel_multiply_factor=config.model.edit_channel_multiply_factor,
            chop_editor_at_layer=config.model.chop_editor_at_layer,
            num_editing_heads=config.model.num_editing_heads,
            use_layerwise_embeddings=config.model.use_layerwise_embeddings,
            edit_dampening_factor=config.model.edit_dampening_factor,
            kill_token_zero=config.model.kill_token_zero,
            use_ghost_token=config.model.use_ghost_token,
            compute_position_ids=config.model.compute_position_ids,
            cross_attn_layers=list(config.model.cross_attn_layers),
            restrict_edit_to_layers=list(config.model.restrict_edit_to_layers),
            restrict_edit_to_positions=list(config.model.restrict_edit_to_positions),
        )

        model = model_cls(model_config)

        if config.model.target_ckpt:
            target_state_dict = torch.load(
                config.model.target_ckpt, map_location="cpu"
            )["state"]
            model.load_target_model(target_state_dict)

    elif config.mode == "finetune_sft":
        model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path)

    if config.mode in ["train_editor", "finetune_sft"]:
        train_dataset = get_task(config, config.task.name, config.data.train_split_name)
        train_dataloader = get_dataloader(
            train_dataset, config, config.data.train_split_name
        )

        if config.train.do_eval:
            validation_dataset = get_task(
                config, config.task.name, config.data.val_split_name
            )
            validation_dataloader = get_dataloader(
                validation_dataset, config, config.data.val_split_name
            )
        else:
            validation_dataloader = None

        # print trainable params
        trainable_params, all_params = get_nb_trainable_parameters(model)
        logger.info(
            f"trainable/total params: {trainable_params} / {all_params} ({100 *trainable_params/all_params:.3f}%)"
        )

        # set experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # create timestamp for use in exp_name
        config.exp_name = f"{config.exp_name}_{timestamp}"

        train_fn = train if config.mode == "train_editor" else finetune

        if config.train.use_ddp:
            world_size = torch.cuda.device_count()
            mp.spawn(
                train_fn,
                nprocs=world_size,
                args=(
                    world_size,
                    config,
                    model,
                    train_dataloader,
                    validation_dataloader,
                ),
            )
        else:
            train_fn(0, 1, config, model, train_dataloader, validation_dataloader)
    elif config.mode == "eval":
        if config.data.padding_side == "right":
            logger.warning(
                "You are using right padding, which is not supported for evaluation. "
                "Switching to left padding."
            )
            config.data.padding_side = "left"
        eval_dataset = get_task(config, config.task.name, config.data.val_split_name)
        eval_dataloader = get_dataloader(
            eval_dataset, config, config.data.val_split_name
        )
        evaluate(config, model, eval_dataloader)


if __name__ == "__main__":
    main()
