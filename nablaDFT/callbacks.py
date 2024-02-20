import os
import warnings

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SaveModel(Callback):
    """Save best model weights to specified location"""

    def __init__(self, dirpath, filename):
        super().__init__()
        self.filename = filename
        self.dirpath = dirpath
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath, exist_ok=True)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        save_model(trainer, os.path.join(self.dirpath, self.filename))


def save_model(trainer, filepath):
    """Saves only model weights"""
    if trainer.checkpoint_callback is not None:
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
    else:
        warnings.warn(
            "Checkpoint callback was not specified, skip saving best model weights"
        )
        return
    best_model = trainer.model.__class__.load_from_checkpoint(best_ckpt_path).to_cpu()
    best_model.eval()
    torch.save(best_model.net, filepath)
