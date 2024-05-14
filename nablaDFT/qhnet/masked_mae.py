import torch
from torch import Tensor

from torchmetrics.functional.regression.mae import _mean_absolute_error_update
from torchmetrics.regression import MeanAbsoluteError


class MaskedMeanAbsoluteError(MeanAbsoluteError):
    """Overloaded MAE for usage with block diagonal matrix.
    Mask calculated from target tensor."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Both inputs are block diagonal Torch Tensors."""
        sum_abs_error, _ = _mean_absolute_error_update(preds, target)
        num_obs = torch.count_nonzero(target).item()

        self.sum_abs_error += sum_abs_error
        self.total += num_obs
