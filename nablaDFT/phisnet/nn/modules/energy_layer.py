import torch
import torch.nn as nn

"""
FCN for energy prediction
"""


class EnergyLayer(nn.Module):
    def __init__(self, num_in, num_out, activation, zero_init=False):
        super(EnergyLayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.zero_init = zero_init
        self.linear_diagonal = nn.Linear(self.num_in, self.num_out)
        self.linear_offdiagonal = nn.Linear(self.num_in, self.num_out)
        self.linear_out = nn.Linear(2 * self.num_out, 1)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.zero_init:
            nn.init.zeros_(self.linear_diagonal.weight)
            nn.init.zeros_(self.linear_offdiagonal.weight)
            nn.init.zeros_(self.linear_out.weight)
        else:
            nn.init.orthogonal_(self.linear_diagonal.weight)
            nn.init.orthogonal_(self.linear_offdiagonal.weight)
            nn.init.orthogonal_(self.linear_out.weight)

        nn.init.zeros_(self.linear_diagonal.bias)
        nn.init.zeros_(self.linear_offdiagonal.bias)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, fii, fij, sizes, pair_sizes):
        fii_ = self.activation(self.linear_diagonal(fii[0].squeeze(dim=-1))).squeeze(dim=2)

        diagonal_features = torch.concat(
            [torch.mean(sample_features, dim=1) for sample_features in torch.split(fii_, sizes, dim=1)])
        fij_ = self.activation(self.linear_offdiagonal(fij[0].squeeze(dim=-1))).squeeze(dim=2)

        offdiagonal_features = torch.concat(
            [torch.mean(sample_features, dim=1) for sample_features in torch.split(fij_, pair_sizes, dim=1)])

        full_features = torch.concat([diagonal_features, offdiagonal_features], dim=1)

        energies = self.linear_out(full_features)
        return energies
