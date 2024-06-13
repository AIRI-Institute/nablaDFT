import glob
import os
import json
import random
from typing import List
from urllib import request as request
from pathlib import Path
import logging
import warnings

from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.ddp import DDPStrategy

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict
import wandb
import torch


logger = logging.getLogger()


def get_file_size(url: str) -> int:
    """Returns file size in bytes
    
    Args:
        url (str): url of file to download
    """
    req = request.Request(url, method="HEAD")
    with request.urlopen(req) as f:
        file_size = f.headers.get('Content-Length')
    return int(file_size)


def tqdm_download_hook(t):
    """wraps TQDM progress bar instance"""
    last_block = [0]

    def update_to(blocks_count: int, block_size: int, total_size: int):
        """Adds progress bar for request.urlretrieve() method
        Args:
            - blocks_count (int): transferred blocks count.
            - block_size (int): size of block in bytes.
            - total_size (int): size of requested file.
        """
        if total_size in (None, -1):
            t.total = total_size
        displayed = t.update((blocks_count - last_block[0]) * block_size)
        last_block[0] = blocks_count
        return displayed
    return update_to


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


def load_model(config: DictConfig, ckpt_path: str) -> LightningModule:
    """Instantiates model and loads model weights from checkpoint.

    Args:
        config (DictConfig): config for task. see r'config/' for examples.
        ckpt_path (str): path to checkpoint.
    """
    model: LightningModule = hydra.utils.instantiate(config.model)
    if ckpt_path is None:
        warnings.warn(
            """Checkpoint path was specified, but it not exists. Continue with randomly initialized weights."""
        )
    else:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Restore model weights from {ckpt_path}")
    model.eval()
    return model


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
