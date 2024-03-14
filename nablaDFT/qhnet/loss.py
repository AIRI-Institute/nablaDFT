import torch
import torch.nn as nn
import torch.nn.functional as F


class HamiltonianLoss(nn.Module):
    def __init__(self) -> None:
        super(HamiltonianLoss, self).__init__()

    def forward(self, pred, target, mask):
        diff = pred - target
        mse = torch.mean(diff**2)
        mae = torch.mean(torch.abs(diff))
        mse *= (pred.numel() / mask.sum())
        mae *= (pred.numel() / mask.sum())
        rmse = torch.sqrt(mse)
        return rmse + mae
