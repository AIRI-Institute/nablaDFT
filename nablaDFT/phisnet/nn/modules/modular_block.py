import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_stack import *
from .interaction_block import *

"""
Basic building block of the neural network which refines atomic features in an iterative way
"""
class ModularBlock(nn.Module):
    def __init__(self, order, num_features, num_basis_functions, num_residual_pre_x,  num_residual_post_x, 
            num_residual_pre_vi, num_residual_pre_vj, num_residual_post_v, num_residual_output, 
            clebsch_gordan=None, mix_orders=True, activation='swish'):
        super(ModularBlock, self).__init__()
        #initialize attributes
        self.order               = order
        self.num_features        = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_pre_x  = num_residual_pre_x
        self.num_residual_post_x = num_residual_post_x
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.num_residual_output = num_residual_output
        #initialize modules
        self.interaction = InteractionBlock(self.order, self.num_features, self.num_basis_functions, 
                self.num_residual_pre_vi, self.num_residual_pre_vj, self.num_residual_post_v, 
                clebsch_gordan, mix_orders, activation)
        self.residual_pre_x = ResidualStack(self.num_residual_pre_x, self.order, self.num_features, 
                clebsch_gordan, mix_orders, activation)
        self.residual_post_x = ResidualStack(self.num_residual_post_x, self.order, self.num_features, 
                clebsch_gordan, mix_orders, activation)
        self.residual_out = ResidualStack(self.num_residual_output, self.order, self.num_features, 
                clebsch_gordan, mix_orders, activation)

    def forward(self, xs, rbf, sph, idx_i, idx_j):
        xs = self.residual_pre_x(xs)
        xs = self.interaction(xs, rbf, sph, idx_i, idx_j)
        xs = self.residual_post_x(xs)
        ys = self.residual_out(xs)
        return xs, ys




