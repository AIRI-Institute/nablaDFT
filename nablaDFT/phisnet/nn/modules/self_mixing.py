import torch
import torch.nn as nn
import numpy as np

"""
Mixes features of different orders
"""
class SelfMixing(nn.Module):
    def __init__(self, order_in, order_out, num_features, clebsch_gordan):
        super(SelfMixing, self).__init__()
        self.order_in       = order_in
        self.order_out      = order_out
        self.num_features   = num_features
        self.clebsch_gordan = clebsch_gordan
        #coefficients for mixing
        for l1 in range(self.order_in+1):
            for l2 in range(l1+1, self.order_in+1):
                for L in range(abs(l1-l2),min(l1+l2,self.order_out)+1):
                    name = 'mixcoeff_{}_{}_{}'.format(l1, l2, L)
                    self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
        for L in range(min(self.order_in, self.order_out)+1):
            name = 'keepcoeff_{}'.format(L)
            self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        count = [0 for L in range(self.order_out+1)]
        for L in range(min(self.order_in, self.order_out)+1):
            count[L] += 1
        for l1 in range(self.order_in+1):
            for l2 in range(l1+1, self.order_in+1):
                for L in range(abs(l1-l2),min(l1+l2,self.order_out)+1):
                    count[L] += 1

        for L in range(min(self.order_in, self.order_out)+1):
            nn.init.uniform_(self.keepcoeff(L), a=-np.sqrt(3/count[L]), b=np.sqrt(3/count[L]))

        for l1 in range(self.order_in+1):
            for l2 in range(l1+1, self.order_in+1):
                for L in range(abs(l1-l2),min(l1+l2,self.order_out)+1):
                    nn.init.uniform_(self.mixcoeff(l1, l2, L), a=-np.sqrt(3/count[L]), b=np.sqrt(3/count[L]))

    def keepcoeff(self, L):
        return getattr(self, 'keepcoeff_{}'.format(L))

    def mixcoeff(self, l1, l2, L):
        return getattr(self, 'mixcoeff_{}_{}_{}'.format(l1,l2,L))

    def forward(self, xs):
        #initialize output
        ys = [self.keepcoeff(L)*xs[L] if L <= self.order_in 
              else torch.zeros_like(xs[0]).repeat(*(1,)*len(xs[0].shape[:-2]),2*L+1,1) 
              for L in range(self.order_out+1)]
        #loop over all combinations of orders
        for l1 in range(self.order_in+1):
            #get view of x[l1] that enables broadcasting to compute the spherical tensor product
            x1 = xs[l1].view(*xs[l1].shape[:-2], xs[l1].size(-2), 1, 1, self.num_features)
            for l2 in range(l1+1, self.order_in+1):
                #get view of x[l2] that enables broadcasting to compute the spherical tensor product
                x2 = xs[l2].view(*xs[l2].shape[:-2], 1, xs[l2].size(-2), 1, self.num_features)
                #compute spherical tensor product
                tp = x1*x2
                #decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1-l2), min(l1+l2, self.order_out)+1):
                    #get Clebsch-Gordan coefficients in broadcastable form
                    cg = self.clebsch_gordan(l1, l2, L)
                    cg = cg.view((*(1,)*len(tp.shape[:-4]), *cg.shape, 1))
                    #get coefficients in broadcastable form
                    coeff = self.mixcoeff(l1, l2, L).view(*(1,)*len(tp.shape[:-4]), 1, -1)
                    #contract and add
                    ys[L] = ys[L] + coeff*((cg*tp).sum(-3).sum(-3))
        return ys

