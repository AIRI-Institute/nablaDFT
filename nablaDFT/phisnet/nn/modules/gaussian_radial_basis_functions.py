import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import cutoff_function, softplus_inverse

"""
computes radial basis functions with exponential Gaussians
"""
class GaussianRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff):
        super(GaussianRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('center', torch.linspace(0, cutoff, self.num_basis_functions, dtype=torch.float64))
        self.register_buffer('width', torch.tensor(self.num_basis_functions/cutoff, dtype=torch.float64))
        #for compatibility with other basis functions on tensorboard, doesn't do anything
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r):
        rbf = cutoff_function(r, self.cutoff) * torch.exp(-self.width*(r-self.center)**2)
        return rbf 


