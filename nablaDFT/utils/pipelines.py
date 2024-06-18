import glob
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from ase.db import connect
from dotenv import load_dotenv
from omegaconf import DictConfig, open_dict
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

JOB_TYPES = ["train", "test", "predict"]
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
    if config.job_type == "optimize":
        return config

    datamodule_cls = config.datamodule._target_
    if datamodule_cls == "nablaDFT.dataset.ASENablaDFT":
        config.root = os.path.join(config.root, "raw")
    if config.name in ["SchNet", "PaiNN", "Dimenet++"]:
        with open_dict(config):
            config.trainer.inference_mode = False
    if len(config.devices) > 1:
        with open_dict(config):
            config.trainer.strategy = DDPStrategy()
    if config.job_type == "train" and config.name == "QHNet":
        with open_dict(config):
            config.trainer.find_unused_parameters = True
    return config


def check_cfg_parameters(cfg: DictConfig):
    job_type = cfg.get("job_type")
    if job_type not in JOB_TYPES:
        raise ValueError(f"job_type must be one of {JOB_TYPES}, got {job_type}")
    if cfg.pretrained and cfg.ckpt_path:
        raise ValueError(
            "Config parameters ckpt_path and pretrained are mutually exclusive. Consider set ckpt_path "
            "to null, if you plan to use pretrained checkpoints."
        )


def write_predictions_to_db(input_db_path: Path, output_db_path: Path, predictions: List[torch.Tensor]):
    if not output_db_path.parent.exists():
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
    if output_db_path.exists():
        output_db_path.unlink()
    if isinstance(predictions[0], Tuple):
        energy = torch.cat([x[0].detach() for x in predictions], dim=0)
        forces = torch.cat([x[1].detach() for x in predictions], dim=0)
    elif isinstance(predictions[0], Dict):
        energy = torch.cat([x["energy"].detach() for x in predictions], dim=0)
        forces = torch.cat([x["forces"].detach() for x in predictions], dim=0)
    energy = energy.numpy()
    forces = forces.numpy()
    force_idx = 0
    with connect(str(input_db_path.resolve())) as in_db:
        with connect(str(output_db_path.resolve())) as out_db:
            for idx in range(1, len(in_db) + 1):
                row = in_db.get(idx)
                natoms = row.natoms
                data = row.data
                if not data:
                    data = {}
                data["energy_pred"] = [float(energy[idx - 1])]
                data["forces_pred"] = forces[force_idx : (force_idx + natoms)]
                out_db.write(row, data=data)
    logger.info(f"Write predictions to {output_db_path}")
