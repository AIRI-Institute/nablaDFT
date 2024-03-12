import torch
import torch.nn as nn
import torch.nn.functional as F


class MAE_RMSE_Loss(nn.Module):
    def __init__(self) -> None:
        super(MAE_RMSE_Loss, self).__init__()

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction=None)
        mae = F.l1_loss(pred, target, reduction="mean")
        rmse = torch.sqrt(mse.mean())
        return mae + rmse
