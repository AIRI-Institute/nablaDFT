import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    Callback,
    seed_everything,
)

from nablaDFT.utils import close_loggers, load_model
from nablaDFT.dataset import NablaDFT


def run(config: DictConfig):
    if config.get("seed"):
        seed_everything(config.seed)
    job_type = config.get("job_type")
    if config.get("ckpt_path"):
        ckpt_path = os.path.join(
            hydra.utils.get_original_cwd(), config.get("ckpt_path")
        )
    else:
        ckpt_path = None
    if job_type == "test" and ckpt_path is not None:
        model = load_model(config, ckpt_path)
    else:
        model: LightningModule = hydra.utils.instantiate(config.model)
    # Callbacks
    callbacks: List[Callback] = []
    for _, callback_cfg in config.callbacks.items():
        callbacks.append(hydra.utils.instantiate(callback_cfg))
    # Loggers
    loggers: List[Logger] = []
    for _, logger_cfg in config.loggers.items():
        loggers.append(hydra.utils.instantiate(logger_cfg))
    # Trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers
    )
    # Datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    if job_type == "train":
        trainer.fit(model=model, datamodule=datamodule.dataset, ckpt_path=ckpt_path)
    else:
        trainer.test(model=model, datamodule=datamodule.dataset, ckpt_path=ckpt_path)

    # Finalize
    close_loggers(
        logger=loggers,
    )
