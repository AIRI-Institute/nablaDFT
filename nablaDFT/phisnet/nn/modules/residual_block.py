import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherical_linear import SphericalLinear
from .shifted_softplus import ShiftedSoftplus
from .swish import Swish

"""
Pre-activation residual block
"""
class ResidualBlock(nn.Module):
    def __init__(self, order, num_features, clebsch_gordan=None, mix_orders=True, activation='swish'):
        super(ResidualBlock, self).__init__()
        self.order = order
        self.num_features = num_features
        self.mix_orders = mix_orders
        if self.mix_orders: 
            assert clebsch_gordan is not None
        if activation == 'swish':
            self.activation_pre  = Swish(self.num_features)
            self.activation_post = Swish(self.num_features)
        elif activation == 'ssp':
            self.activation_pre  = ShiftedSoftplus(self.num_features)
            self.activation_post = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", activation)
            quit()
        self.linear1 = SphericalLinear(self.order, self.num_features, self.order, self.num_features,
            clebsch_gordan, self.mix_orders)
        self.linear2 = SphericalLinear(self.order, self.num_features, self.order, self.num_features,
            clebsch_gordan, self.mix_orders, zero_init=True)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, xs):
        ys = [1*x for x in xs]
        ys[0] = self.activation_pre(ys[0])
        ys = self.linear1(ys)
        ys[0] = self.activation_post(ys[0])
        ys = self.linear2(ys)
        return [x+y for x,y in zip(xs,ys)]


