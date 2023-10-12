import torch
import torch.nn as nn


def l2loss_atomwise(pred, target, reduction='mean'):
    dist = torch.linalg.vector_norm((pred - target), dim=-1)
    if reduction == 'mean':
        return torch.mean(dist)
    elif reduction == 'sum':
        return torch.sum(dist)
    else:
        return dist


class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = l2loss_atomwise(pred, target, self.reduction)
        return loss
