from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
import schnetpack as spk
import torch
from schnetpack.model.base import AtomisticModel
from schnetpack.task import UnsupervisedModelOutput
from torch import nn


class AtomisticTaskFixed(spk.task.AtomisticTask):
    def __init__(
        self,
        model_name: str,
        model: AtomisticModel,
        outputs: List[spk.task.ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        pl.LightningModule.__init__(self)
        self.model_name = model_name
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=["model"])

    def predict_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass
        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)
        return pred

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # reshape model.postprocessors (AddOffsets)
        #  otherwise during test error will occur
        checkpoint["state_dict"]["model.postprocessors.0.mean"] = checkpoint[
            "state_dict"
        ]["model.postprocessors.0.mean"].reshape(1)

    # override base class method
    def predict_without_postprocessing(self, batch):
        pred = self(batch)
        return pred
