import random
from datetime import datetime

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from data import get_dataloader, get_task
from models.gpt2 import GPT2Editor, GPT2EditorConfig
from train_utils import train

#should I need to do this? 
#trying to solve:
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

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
    )
    editor_model = GPT2Editor(model_config)

    if config.mode == "train":
        train_dataloader = get_dataloader(
            get_task(config, config.task.name, "train"), config, "train"
        )
        validation_dataloader = get_dataloader(
            get_task(config, config.task.name, "val"), config, "val"
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
