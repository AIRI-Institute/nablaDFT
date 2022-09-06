import torch
import torch.nn as nn
from .self_mixing import SelfMixing

"""
Like a linear layer, but acting on spherical harmonic features (optionally mixes features)
"""
class SphericalLinear(nn.Module):
    def __init__(self, order_in, num_in, order_out, num_out, clebsch_gordan=None, mix_orders=True, bias=True, zero_init=False):
        super(SphericalLinear, self).__init__()
        self.order_in = order_in
        self.num_in = num_in
        self.order_out = order_out
        self.num_out = num_out
        self.bias = bias
        self.mix_orders = mix_orders
        self.zero_init = zero_init
        if self.mix_orders: 
            assert clebsch_gordan is not None #Clebsch-Gordan coefficients are necessary for mixing
            self.mixing = SelfMixing(self.order_in, self.order_out, self.num_in, clebsch_gordan)
        else: #order can only be changed if mixing is enabled
            assert order_in == order_out
        self.linear = nn.ModuleList([nn.Linear(self.num_in, self.num_out, 
            bias=(self.bias and L == 0)) for L in range(self.order_out+1)])
        self.reset_parameters()

    def reset_parameters(self):
        if self.zero_init:
            for L in range(self.order_out+1):
                nn.init.zeros_(self.linear[L].weight)
        else:
            for L in range(self.order_out+1):
                nn.init.orthogonal_(self.linear[L].weight)
        if self.bias:
            nn.init.zeros_(self.linear[0].bias)

    def forward(self, xs):
        if self.mix_orders:
            ys = self.mixing(xs)
            for L in range(self.order_out+1):
                ys[L] = self.linear[L](ys[L])
        else:
            ys = []
            for x, linear in zip(xs, self.linear):
                ys.append(linear(x))
        return ys

