import os
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn

"""
Helper class that stores Clebsch-Gordan coefficients
"""


class ClebschGordan(nn.Module):
    def __init__(self):
        super(ClebschGordan, self).__init__()
        tmp = np.load(
            os.path.join(os.path.dirname(__file__), "clebsch_gordan_coefficients_L10.npz"),
            allow_pickle=True,
        )["cg"][()]
        # add permutations (the npz file only stores coefficients for l1 <= l2 <= l3) and register buffers
        for l123 in tmp.keys():
            for a, b, c in permutations((0, 1, 2)):
                name = "cg_{}_{}_{}".format(l123[a], l123[b], l123[c])
                if name not in dir(self):
                    self.register_buffer(name, torch.tensor(tmp[l123].transpose(a, b, c)))

    def forward(self, l1, l2, l3):
        return getattr(self, "cg_{}_{}_{}".format(l1, l2, l3))
