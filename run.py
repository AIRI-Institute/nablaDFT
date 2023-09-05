"""Usage: python run.py --config-name path-to-config"""

import hydra
from omegaconf import DictConfig

from nablaDFT.utils import load_envs, init_wandb
from nablaDFT.pipelines import run


@hydra.main(
    config_path="./config",
    config_name=None,
    version_base="1.2"
)
def main(config: DictConfig):
    load_envs()
    init_wandb()
    run(config)


if __name__ == "__main__":
    main()
