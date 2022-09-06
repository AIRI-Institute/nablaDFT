import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import cutoff_function, softplus_inverse

"""
computes radial basis functions with exponential Gaussians
"""
class ExponentialGaussianRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(ExponentialGaussianRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('center', torch.linspace(1, 0, self.num_basis_functions, dtype=torch.float64))
        self.register_buffer('width', torch.tensor(1.0*self.num_basis_functions, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha = F.softplus(self._alpha)
        rbf = cutoff_function(r, self.cutoff) * torch.exp(-self.width*(torch.exp(-alpha*r)-self.center)**2)
        return rbf 


