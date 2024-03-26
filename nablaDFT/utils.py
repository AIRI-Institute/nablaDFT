import glob
import os
import json
import random
from typing import List
from urllib import request as request
import logging
import warnings

from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig
import wandb
import torch
import hydra


import nablaDFT


logger = logging.getLogger()

def get_file_size(url: str) -> int:
    """Returns file size in bytes"""
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


def download_model(config: DictConfig) -> str:
    """Downloads best model checkpoint from vault."""
    model_name = config.get("name")
    ckpt_path = os.path.join(
        hydra.utils.get_original_cwd(),
        f"./checkpoints/{model_name}/{model_name}_100k.ckpt",
    )
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        os.makedirs(f"./checkpoints/{model_name}", exist_ok=True)
    with open(nablaDFT.__path__[0] + "/links/models_checkpoints.json", "r") as f:
        data = json.load(f)
        url = data[f"{model_name}"]["dataset_train_100k"]
    file_size = get_file_size(url)
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, total=file_size, desc=f"Downloading {model_name} checkpoint") as t:
        request.urlretrieve(url, ckpt_path, reporthook=tqdm_download_hook(t))
    logging.info(f"Downloaded {model_name} 100k checkpoint to {ckpt_path}")
    return ckpt_path


def load_model(config: DictConfig, ckpt_path: str) -> LightningModule:
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
