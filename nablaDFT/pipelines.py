import os
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from nablaDFT import model_registry
from nablaDFT.dataset import ASENablaDFT
from nablaDFT.optimization import BatchwiseOptimizeTask
from nablaDFT.utils import (
    check_cfg_parameters,
    close_loggers,
    load_from_checkpoint,
    seed_everything,
    set_additional_params,
    write_predictions_to_db,
)


def predict(
    trainer: Trainer, model: LightningModule, datamodule: LightningDataModule, model_name: str, output_dir: str
):
    """Function for prediction loop execution.

    Saves model prediction to "predictions" directory.
    """
    trainer.logger = False  # need this to disable save_hyperparameters during predict,
    # otherwise OmegaConf DictConf can't be dumped to YAML
    output_db_path = Path(output_dir) / f"{model_name}_{datamodule.dataset_name}.db"
    if isinstance(datamodule, ASENablaDFT):
        input_db_path = Path(datamodule.datapath).parent / f"raw/{datamodule.dataset_name}.db"
    else:
        input_db_path = Path(datamodule.root) / f"raw/{datamodule.dataset_name}.db"
    predictions = trainer.predict(model=model, datamodule=datamodule)
    write_predictions_to_db(input_db_path, output_db_path, predictions)


def optimize(config: DictConfig):
    """Function for batched molecules optimization.
    Uses model defined in config.

    Args:
        ckpt_path (str): path to model checkpoint.
        config (DictConfig): config for task. see r'config/' for examples.
    """
    output_dir = config.get("output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_datapath = os.path.join(output_dir, f"{config.name}_{config.dataset_name}.db")
    # append to existing database not supported
    if os.path.exists(output_datapath):
        os.remove(output_datapath)
    model = load_from_checkpoint(config)
    calculator = hydra.utils.instantiate(config.calculator, model)
    optimizer = hydra.utils.instantiate(config.optimizer, calculator)
    task = BatchwiseOptimizeTask(
        config.input_db,
        output_datapath,
        optimizer=optimizer,
        batch_size=config.batch_size,
    )
    task.run()


def run(config: DictConfig):
    """Main function to perform task runs on nablaDFT datasets.
    Refer to r'nablaDFT/README.md' for detailed description of run configuration.

    Args:
        config (DictConfig): config for task. see r'config/' for examples.
    """
    check_cfg_parameters(config)
    if config.get("seed"):
        seed_everything(config.seed)
    job_type = config.job_type
    if config.get("ckpt_path"):
        config.ckpt_path = Path(hydra.utils.get_original_cwd()) / config.get("ckpt_path")
    else:
        config.ckpt_path = None
    config = set_additional_params(config)
    # download checkpoint if pretrained=True
    if job_type == "optimize":
        return optimize(config)
    if config.get("pretrained"):
        model: LightningModule = model_registry.get_pretrained_model("lightning", config.pretrained)
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
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    # Datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    if job_type == "train":
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    elif job_type == "test":
        trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    elif job_type == "predict":
        predict(trainer, model, datamodule, config.model.model_name, config.output_dir)
    # Finalize
    close_loggers(
        logger=loggers,
    )
