import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherical_linear import *
from .residual_stack import *
from .pair_mixing import *

"""
Refines atomic features by interacting with its neighbors
"""
class InteractionBlock(nn.Module):
    def __init__(self, order, num_features, num_basis_functions, num_residual_pre_vi, num_residual_pre_vj, 
            num_residual_post_v, clebsch_gordan=None, mix_orders=True, activation='swish'):
        super(InteractionBlock, self).__init__()
        #initialiye attributes
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        #initialize activation function
        if activation == 'swish':
            self.activation_i = Swish(self.num_features)
            self.activation_j = Swish(self.num_features)
            self.activation_v = Swish(self.num_features)
        elif activation == 'ssp':
            self.activation_i = ShiftedSoftplus(self.num_features)
            self.activation_j = ShiftedSoftplus(self.num_features)
            self.activation_v = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", activation)
            quit()
        #initialize modules
        self.angular_fn1 = SphericalLinear(self.order, 1, self.order, self.num_features, clebsch_gordan, mix_orders=False)
        self.angular_fn2 = SphericalLinear(self.order, 1, self.order, self.num_features, clebsch_gordan, mix_orders=False)
        self.radial_fn = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.mixing = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, clebsch_gordan)
        self.linear_i = SphericalLinear(self.order, self.num_features, self.order, self.num_features, clebsch_gordan, mix_orders)
        self.linear_j = SphericalLinear(self.order, self.num_features, self.order, self.num_features, clebsch_gordan, mix_orders)
        self.linear_v = SphericalLinear(self.order, self.num_features, self.order, self.num_features, clebsch_gordan, mix_orders)
        self.residual_pre_vi = ResidualStack(self.num_residual_pre_vi, self.order, self.num_features, clebsch_gordan, mix_orders, activation)
        self.residual_pre_vj = ResidualStack(self.num_residual_pre_vj, self.order, self.num_features, clebsch_gordan, mix_orders, activation)
        self.residual_post_v = ResidualStack(self.num_residual_post_v, self.order, self.num_features, clebsch_gordan, mix_orders, activation)
        self.reset_parameters()

    def reset_parameters(self):
        for L in range(self.order+1):
            nn.init.orthogonal_(self.radial_fn[L].weight)

    def forward(self, xs, rbf, sph, idx_i, idx_j):
        ys = [1*x for x in xs]
        #path for atoms i
        yi = self.residual_pre_vi(ys)
        yi[0] = self.activation_i(yi[0])
        yi = self.linear_i(yi)
        #path for atoms j
        yj = self.residual_pre_vj(ys)
        yj[0] = self.activation_j(yj[0])
        yj = self.linear_j(yj)
        #interaction function
        for L in range(self.order+1):
            idx = idx_j.view(*(1,)*len(yj[L].shape[:-3]),-1,1,1).repeat(*yj[L].shape[:-3], 1, *yj[L].shape[-2:])
            yj[L] = torch.gather(yj[L], 1, idx)
        vs = self.mixing(yj, self.angular_fn1(sph), rbf)
        a = self.angular_fn2(sph)
        for L in range(self.order+1):
            vs[L] = yi[L].index_add(1, idx_i, vs[L]+self.radial_fn[L](rbf)*a[L]*yj[0])
        #interaction refinement
        vs = self.residual_post_v(vs)
        vs[0] = self.activation_v(vs[0])
        vs = self.linear_v(vs)
        return [x+v for x,v in zip(xs,vs)]



