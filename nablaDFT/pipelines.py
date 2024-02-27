import os
from typing import List

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning.loggers import Logger
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    Callback,
    seed_everything,
)

from nablaDFT.utils import close_loggers, load_model
from nablaDFT.optimization import BatchwiseOptimizeTask


JOB_TYPES = ['train', 'test', 'predict', 'optimize']


def predict(trainer: Trainer,
            model: LightningModule,
            datamodule: LightningDataModule,
            ckpt_path: str,
            config: DictConfig
        ):
    """Function for prediction loop execution.
       Saves model prediction to "predictions" directory.
    """
    trainer.logger = False # need this to disable save_hyperparameters during predict, otherwise OmegaConf DictConf can't be dumped to YAML
    pred_path = os.path.join(os.getcwd(), "predictions")
    os.makedirs(pred_path, exist_ok=True)
    predictions = trainer.predict(model=model, datamodule=datamodule.dataset, ckpt_path=ckpt_path)
    torch.save(predictions, f"{pred_path}/{config.name}_{config.dataset_name}.pt")


def optimize(config: DictConfig):
    """Function for batched molecules optimization.
       Uses model defined in config.
    """
    if config.get("ckpt_path"):
        ckpt_path = os.path.join(
            hydra.utils.get_original_cwd(), config.get("ckpt_path")
        )
    else:
        raise RuntimeError(
            f"Job type is optimize, but ckpt_path is None. Specify checkpoint path in model config"
        )
    output_dir = config.get("output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_datapath = os.path.join(output_dir, f"{config.name}_{config.dataset_name}.db")
    model = load_model(config, ckpt_path)
    calculator = hydra.utils.instantiate(
        config.calculator, model 
    )
    optimizer = hydra.utils.instantiate(
        config.optimizer, calculator 
    )
    task = BatchwiseOptimizeTask(
        config.input_db, output_datapath,
        optimizer=optimizer, batch_size=config.batch_size
    )
    task.run()


def run(config: DictConfig):
    if config.get("seed"):
        seed_everything(config.seed)
    job_type = config.get("job_type")
    if job_type not in JOB_TYPES:
        raise ValueError(f"job_type must be one of {JOB_TYPES}, got {job_type}")
    if job_type == "optimize":
        return optimize(config)

    if config.get("ckpt_path"):
        ckpt_path = os.path.join(
            hydra.utils.get_original_cwd(), config.get("ckpt_path")
        )
    else:
        ckpt_path = None
    if job_type in ["test", "predict"]:
        if ckpt_path is None:
            raise RuntimeError(
                f"Job type is {job_type}, but ckpt_path is {ckpt_path}. Specify checkpoint path in model config"
            )
        else:
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
    if job_type == 'train':
        trainer.fit(model=model, datamodule=datamodule.dataset, ckpt_path=ckpt_path)
    elif job_type == 'test':
        trainer.test(model=model, datamodule=datamodule.dataset, ckpt_path=ckpt_path)
    elif job_type == 'predict':
        predict(trainer, model, datamodule, ckpt_path, config)
    # Finalize
    close_loggers(
        logger=loggers,
    )

