import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_block import *

"""
Stack of pre-activation residual blocks
"""
class ResidualStack(nn.Module):
    def __init__(self, num_blocks, order, num_features, clebsch_gordan=None, mix_orders=True, activation='swish'):
        super(ResidualStack, self).__init__()
        self.num_blocks = num_blocks
        self.order = order
        self.num_features = num_features
        self.stack = nn.ModuleList([ResidualBlock(self.order, self.num_features, 
            clebsch_gordan, mix_orders, activation) for i in range(self.num_blocks)])

    def forward(self, xs):
        if self.num_blocks > 0:
            for residual_block in self.stack:
                xs = residual_block(xs)
            return xs
        else: #to prevent inplace modification
            return [1*x for x in xs]



