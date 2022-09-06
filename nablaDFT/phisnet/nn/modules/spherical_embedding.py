import math
import numpy as np
import torch
import torch.nn as nn
from .embedding import *

"""
Wraps the ordinary embedding layer and returns features for all orders
"""
class SphericalEmbedding(nn.Module):
    def __init__(self, order, num_features, Zmax=87):
        super(SphericalEmbedding, self).__init__()
        self.order = order
        self.num_features = num_features
        self.Zmax = Zmax
        self.embedding = Embedding(self.num_features, self.Zmax)

    def forward(self, Z):
        xs = []
        for L in range(self.order+1):
            if L == 0:
                xs.append(self.embedding(Z).view(*Z.shape,1,-1).repeat(*(1,)*len(Z.shape),1,1))
            else: #features for L>0 must be zero for rotational invariance
                xs.append(torch.zeros_like(xs[0]).repeat(*(1,)*len(Z.shape),2*L+1,1))
        return xs
