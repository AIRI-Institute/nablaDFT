import glob
import os
import random
from typing import List
import logging
import warnings

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.ddp import DDPStrategy

import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict
import wandb
import torch


logger = logging.getLogger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def close_loggers(
    logger: List,
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def load_envs():
    envs = glob.glob("*.env")
    for env in envs:
        load_dotenv(env)


@rank_zero_only
def init_wandb():
    wandb.login()


def set_additional_params(config: DictConfig) -> DictConfig:
    datamodule_cls = config.datamodule._target_
    if datamodule_cls == "nablaDFT.dataset.ASENablaDFT":
        config.root = os.path.join(config.root, "raw")
    if config.name in ['SchNet', 'PaiNN', 'Dimenet++']:
        with open_dict(config):
            config.trainer.inference_mode = False
    if len(config.devices) > 1:
        with open_dict(config):
            config.trainer.strategy = DDPStrategy()
    if config.job_type == "train" and config.name == "QHNet":
        with open_dict(config):
            config.trainer.find_unused_parameters = True
    return config


def write_predictions_to_db():
    # TODO: write
    raise NotImplementedError
