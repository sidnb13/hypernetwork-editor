import torch
import random
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
