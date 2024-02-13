from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
import schnetpack as spk
import torch
from schnetpack.model.base import AtomisticModel
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
