import random
from datetime import datetime

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from data import get_dataloader, get_task
from helpers import get_nb_trainable_parameters
from logger import get_logger
from models.gpt2 import GPT2Editor, GPT2EditorConfig
from train_utils import train

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

    model_config = GPT2EditorConfig(
        name_or_path=config.model,
        edit_channel_multiply_factor=config.model.edit_channel_multiply_factor,
        chop_editor_at_layer=config.model.chop_editor_at_layer,
        num_editing_heads=config.model.num_editing_heads,
        use_layerwise_embeddings=config.model.use_layerwise_embeddings,
        edit_dampening_factor=config.model.edit_dampening_factor,
        kill_token_zero=config.model.kill_token_zero,
        use_ghost_token=config.model.use_ghost_token,
        compute_position_ids=config.model.compute_position_ids,
        cross_attn_layers=config.model.cross_attn_layers,
        restrict_edit_to_layers=config.model.restrict_edit_to_layers,
        restrict_edit_to_positions=config.model.restrict_edit_to_positions,
    )
    editor_model = GPT2Editor(model_config)

    if config.mode == "train":
        train_dataloader = get_dataloader(
            get_task(config, config.task.name, "train"), config, "train"
        )
        if config.train.do_eval:
            validation_dataloader = get_dataloader(
                get_task(config, config.task.name, "val"), config, "val"
            )
        else:
            validation_dataloader = None

        # print trainable params
        trainable_params, all_params = get_nb_trainable_parameters(editor_model)
        logger.info(
            f"trainable/total params: {trainable_params} / {all_params} ({100 *trainable_params/all_params:.3f}%)"
        )

        # set experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # create timestamp for use in exp_name
        config.exp_name = f"{config.exp_name}_{timestamp}"

        if config.train.use_ddp:
            world_size = torch.cuda.device_count()
            mp.spawn(
                train,
                nprocs=world_size,
                args=(
                    world_size,
                    config,
                    editor_model,
                    train_dataloader,
                    validation_dataloader,
                ),
            )
        else:
            train(0, 1, config, editor_model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
