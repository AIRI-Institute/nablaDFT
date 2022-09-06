import torch
import torch.nn as nn

"""
Learnable swish activation
"""
class Swish(nn.Module):
    def __init__(self, num_features, initial_alpha=1.0, initial_beta=1.702):
        super(Swish, self).__init__()
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
        return self.alpha*x*torch.sigmoid(self.beta*x)