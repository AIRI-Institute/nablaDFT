import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Learnable shifted softplus activation
"""
class ShiftedSoftplus(nn.Module):
    def __init__(self, num_features, initial_alpha=1.0, initial_beta=1.0):
        super(ShiftedSoftplus, self).__init__()
        self._log2         = math.log(2)
        self.num_features  = num_features
        self.initial_alpha = initial_alpha
        self.initial_beta  = initial_beta
        self.register_parameter('alpha', nn.Parameter(torch.Tensor(self.num_features)))
        self.register_parameter('beta',  nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta,  self.initial_beta)

    def forward(self, x):
        return self.alpha*torch.where(self.beta != 0, (F.softplus(self.beta*x) - self._log2)/self.beta, 0.5*x)
