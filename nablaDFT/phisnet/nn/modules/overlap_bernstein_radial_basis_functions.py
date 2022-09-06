import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import binom
from ..functional import cutoff_function, softplus_inverse

"""
computes radial basis functions with overlap Bernstein polynomials
"""
class OverlapBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(OverlapBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        #compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2,num_basis_functions):
            logfactorial[i] = logfactorial[i-1] + np.log(i)
        v = np.arange(0,num_basis_functions)
        n = (num_basis_functions-1)-v
        logbinomial = logfactorial[-1]-logfactorial[v]-logfactorial[n]
        #register buffers and parameters
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha_r = F.softplus(self._alpha)*r
        x = torch.log1p(alpha_r)-alpha_r
        x = self.logc + self.n*x + self.v*torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf 


