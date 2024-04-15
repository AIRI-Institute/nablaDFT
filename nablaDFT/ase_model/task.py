from typing import Any, Dict, List, Optional, Type

import schnetpack as spk
import torch
from schnetpack.model.base import AtomisticModel
from schnetpack.task import UnsupervisedModelOutput


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
        
        super(AtomisticTaskFixed, self).__init__(
            model=model,
            outputs=outputs,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            warmup_steps=warmup_steps
        )
        self.model_name = model_name

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
        if checkpoint["state_dict"].get("model.postprocessors.0.mean", None):
            checkpoint["state_dict"]["model.postprocessors.0.mean"] = checkpoint[
                "state_dict"
            ]["model.postprocessors.0.mean"].reshape(1)
            
    # override base class method
    def predict_without_postprocessing(self, batch):
        pred = self(batch)
        return pred
