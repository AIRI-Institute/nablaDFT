import torch
import torch.nn as nn

"""
Mixes pairs of atomic features in a distance dependent way using a learnable radial function
and outputs pair features
"""
class PairMixing(nn.Module):
    def __init__(self, order_in1, order_in2, order_out, num_basis_functions, num_features, clebsch_gordan):
        super(PairMixing, self).__init__()
        self.order_in1 = order_in1
        self.order_in2 = order_in2
        self.order_out = order_out
        self.num_basis_functions = num_basis_functions
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        #distance-dependent coefficients for mixing
        for l1 in range(self.order_in1+1):
            for l2 in range(self.order_in2+1):
                for L in range(abs(l1-l2), min(l1+l2, self.order_out)+1):
                    name = 'coeff_{}_{}_{}'.format(l1, l2, L)
                    self.add_module(name, nn.Linear(self.num_basis_functions, self.num_features, bias=False))
        self.reset_parameters()

    def reset_parameters(self):
        for l1 in range(self.order_in1+1):
            for l2 in range(self.order_in2+1):
                for L in range(abs(l1-l2),min(l1+l2,self.order_out)+1):
                    nn.init.orthogonal_(self.coeff(l1, l2, L).weight)

    def coeff(self, l1, l2, L):
        return getattr(self, 'coeff_{}_{}_{}'.format(l1, l2, L))

    def forward(self, x1s, x2s, rbf):
        #initialize output to zeros
        ys = [torch.zeros_like(x1s[0]).repeat(*(1,)*len(x1s[0].shape[:-2]),2*L+1,1) 
                for L in range(self.order_out+1)]
        #loop over all combinations of orders
        for l1 in range(self.order_in1+1):
            #get view of x1s[l1] that enables broadcasting to compute the spherical tensor product
            x1 = x1s[l1].view(*x1s[l1].shape[:-2], x1s[l1].size(-2), 1, 1, self.num_features)
            for l2 in range(self.order_in2+1):
                #get view of x2s[l2] that enables broadcasting to compute the spherical tensor product
                x2 = x2s[l2].view(*x2s[l2].shape[:-2], 1, x2s[l2].size(-2), 1, self.num_features)
                #compute spherical tensor product
                tp = x1*x2
                #decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1-l2), min(l1+l2, self.order_out)+1):
                    #get Clebsch-Gordan coefficients in broadcastable form
                    cg = self.clebsch_gordan(l1, l2, L)
                    cg = cg.view((*(1,)*len(tp.shape[:-4]), *cg.shape, 1))
                    #contract and add
                    ys[L] = ys[L] + self.coeff(l1, l2, L)(rbf)*((cg*tp).sum(-3).sum(-3))
        return ys

