# TODO: define base class for all PyG models
import warnings
from typing import Dict

import torch
from torch.optim import Optimizer
import pytorch_lightning as pl


class PyGModel(pl.LightningModule):
    def __init__(self):
        super(PyGModel, self).__init__()

    def predict_step(self, data, **kwargs):
        energy_out, forces_out = self(data)
        return energy_out, forces_out

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        if self.ema is not None:
            self.ema.update()

    def on_fit_start(self) -> None:
        self._instantiate_ema()
        self._check_devices()

    def on_test_start(self) -> None:
        self._instantiate_ema()
        self._check_devices()

    def on_validation_epoch_end(self) -> None:
        self._reduce_metrics(step_type="val")

    def on_test_epoch_end(self) -> None:
        self._reduce_metrics(step_type="test")

    def _calculate_loss(self, y_pred, y_true) -> float:
        total_loss = 0.0
        for name, loss in self.hparams.losses.items():
            total_loss += self.hparams.loss_coefs[name] * loss(y_pred[name], y_true[name])
        return total_loss

    def _calculate_metrics(self, y_pred, y_true) -> Dict:
        """Function for metrics calculation during step."""
        metric = self.hparams.metric(y_pred, y_true)
        return metric

    def _log_current_lr(self) -> None:
        opt = self.optimizers()
        current_lr = opt.optimizer.param_groups[0]["lr"]
        self.log("LR", current_lr, logger=True)

    def _reduce_metrics(self, step_type: str = "train"):
        metric = self.hparams.metric.compute()
        for key in metric.keys():
            self.log(
                f"{step_type}/{key}",
                metric[key],
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.hparams.metric.reset()

    def _check_devices(self):
        self.hparams.metric = self.hparams.metric.to(self.device)
        if self.ema is not None:
            self.ema.to(self.device)

    def _instantiate_ema(self):
        if self.ema is not None:
            self.ema = self.ema(self.parameters())

    def _get_batch_size(self, batch):
        """Function for batch size infer."""
        bsz = batch.batch.max().detach().item() + 1  # get batch size
        return bsz

    def on_fit_end(self):
        if self.global_rank == 0:
            self.save_model("bla-bla")

    def save_model(self, filepath: str):
        """Saves only model weights"""
        if self.trainer.checkpoint_callback is not None:
            best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        else:
            warnings.warn(
                "Checkpoint callback was not specified, skip saving best model weights"
            )
            return
        ckpt = torch.load(best_ckpt_path)
        torch.save(ckpt['net'], filepath)
