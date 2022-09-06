import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import binom
from ..functional import cutoff_function, softplus_inverse

"""
computes radial basis functions with Bernstein polynomials
"""
class BernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff):
        super(BernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
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
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r):
        x = torch.log(r/self.cutoff)
        x = self.logc + self.n*x + self.v*torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf 


