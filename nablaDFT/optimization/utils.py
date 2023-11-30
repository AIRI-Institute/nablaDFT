import glob
import os
import random
from typing import List

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig
import wandb
import torch
import hydra


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
    model: LightningModule = hydra.utils.instantiate(config.model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def save_model(model, path: str):
    print (type(model))
    torch.save(model, path)

