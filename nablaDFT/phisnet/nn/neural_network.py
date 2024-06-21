# coding: utf-8

import math
import os
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .modules import (
    BernsteinRadialBasisFunctions,
    ClebschGordan,
    EnergyLayer,
    ExponentialBernsteinRadialBasisFunctions,
    ExponentialGaussianRadialBasisFunctions,
    GaussianRadialBasisFunctions,
    ModularBlock,
    PairMixing,
    ResidualStack,
    ShiftedSoftplus,
    SphericalEmbedding,
    SphericalLinear,
    Swish,
    clebsch_gordan,
)
from .spherical_harmonics import *


class NeuralNetwork(nn.Module):
    """Neural network for computing Hamiltonian/Overlap matrices in a rotationally equivariant way"""

    def __init__(
        self,
        max_orbitals=None,  # orbitals of atoms, defines layout and shape of output matrix
        order=None,  # 1 maximum order of spherical harmonics features
        num_features=None,  # 32 dimensionality of the feature space
        num_basis_functions=None,  # 32 number of basis functions for featurizing distances
        num_modules=None,  # 1 how many modules are stacked for calculating atomic features (iterations)
        num_residual_pre_x=None,  # 1
        # number of residual blocks applied to atomic features before interaction layer
        num_residual_post_x=None,  # 1
        # number of residual blocks applied to atomic features after interaction layer
        num_residual_pre_vi=None,  # 1
        # number of residual blocks applied to atomic features i before computing interaction features
        num_residual_pre_vj=None,  # 1
        # number of residual blocks applied to atomic features j before computing interaction features
        num_residual_post_v=None,  # 1
        # number of residual blocks applied to interaction features after combining atomic features i/j
        num_residual_output=None,  # 1
        # number of residual blocks applied to atomic features before collecting output atomic features
        num_residual_pc=None,  # 1 number of residual blocks applied to output atomic features for
        # constructing pair features (central atoms)
        num_residual_pn=None,  # 1 number of residual blocks applied to output atomic features for
        # constructing pair features (neighboring atoms)
        num_residual_ii=None,  # 1 number of residual blocks applied to output atomic features for
        # predicting irreps of diagonal blocks (shared)
        num_residual_ij=None,  # 1 number of residual blocks applied to pair features for predicting irreps
        # of off-diagonal blocks (shared)
        num_residual_full_ii=None,  # 1 number of residual blocks applied to output atomic features for
        # predicting irreps of diagonal blocks (full hamiltonian)
        num_residual_full_ij=None,  # 1 number of residual blocks applied to pair features for predicting
        # irreps of off-diagonal blocks (full hamiltonian)
        num_residual_core_ii=None,  # 1 number of residual blocks applied to output atomic features for
        # predicting irreps of diagonal blocks (core hamiltonian)
        num_residual_core_ij=None,  # 1 number of residual blocks applied to pair features for predicting
        # irreps of off-diagonal blocks (core hamiltonian)
        num_residual_over_ij=None,  # 1 number of residual blocks applied to pair features for predicting
        # irreps of off-diagonal blocks (overlap matrix)
        basis_functions=None,  # exp-bernstein
        # type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
        cutoff=None,  # 15.0 cutoff distance (default is 15 Bohr)
        activation=None,  # 'swish' type of activation function used (swish/ssp)
        load_from=None,
        # if this is given the network is loaded from the specified .pth file and all other arguments are
        # ignored
        Zmax=87,
        num_energy_features=64,
        fallback_args: Namespace = None,
    ):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for
        # embeddings, can be kept at default
        super(NeuralNetwork, self).__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian/core_hamiltonian/overlap_matrix/energy/forces?)
        self.calculate_full_hamiltonian = True
        self.calculate_core_hamiltonian = True
        self.calculate_overlap_matrix = True
        self.calculate_energy = False
        self.predict_energy = False
        self.calculate_forces = False
        self.create_graph = True  # can be set to False if the NN is only used for inference

        args = None

        # load state from a file (if load_from is given) and overwrite hyperparameters
        if load_from is not None:
            # 'ID', 'args', 'step', 'epoch', 'best_errors', 'valid_errors', 'model_state_dict',
            # 'optimizer_state_dict', 'scheduler_state_dict', 'exponential_moving_average'
            saved_state = torch.load(load_from, map_location="cpu")

            try:
                args = saved_state["args"]
            except KeyError:
                args = Namespace(**saved_state)

            # print(args)

            max_orbitals = args.max_orbitals if max_orbitals is None else max_orbitals
            order, num_features, num_basis_functions = (
                args.order,
                args.num_features,
                args.num_basis_functions,
            )
            num_modules = args.num_modules

            num_residual_pre_x = args.num_residual_pre_x
            num_residual_post_x = args.num_residual_post_x
            num_residual_pre_vi = args.num_residual_pre_vi
            num_residual_pre_vj = args.num_residual_pre_vj
            num_residual_post_v = args.num_residual_post_v
            num_residual_output = args.num_residual_output

            num_residual_pc, num_residual_pn = (
                args.num_residual_pc,
                args.num_residual_pn,
            )
            num_residual_ii, num_residual_ij = (
                args.num_residual_ii,
                args.num_residual_ij,
            )

            num_residual_full_ii, num_residual_full_ij = (
                args.num_residual_full_ii,
                args.num_residual_full_ij,
            )
            num_residual_core_ii = args.num_residual_core_ii
            num_residual_core_ij = args.num_residual_core_ij
            num_residual_over_ij = args.num_residual_over_ij
            basis_functions, cutoff = args.basis_functions, args.cutoff
            activation = args.activation

        # store hyperparameter values
        self.max_orbitals = max_orbitals
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_modules = num_modules
        self.num_residual_pre_x = num_residual_pre_x
        self.num_residual_post_x = num_residual_post_x
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.num_residual_output = num_residual_output
        self.num_residual_pc = num_residual_pc
        self.num_residual_pn = num_residual_pn
        self.num_residual_ii = num_residual_ii
        self.num_residual_ij = num_residual_ij
        self.num_residual_full_ii = num_residual_full_ii
        self.num_residual_full_ij = num_residual_full_ij
        self.num_residual_core_ii = num_residual_core_ii
        self.num_residual_core_ij = num_residual_core_ij
        self.num_residual_over_ij = num_residual_over_ij
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.activation = activation
        self.Zmax = Zmax
        self.idx_pdict = np.load(
            os.path.join(os.path.dirname(__file__), "modules/pindex_dict.npy"),
            allow_pickle=True,
        ).item()
        self.num_energy_features = num_energy_features

        #  extract nuclear charges from orbitals, determine maximum order, and
        #  build the occupation mask (for extracting occupied orbitals in energy prediction)
        Zl = []
        order_max = 0

        for i in range(len(self.max_orbitals)):
            Zl.append(self.max_orbitals[i][0][0])
            for z, l in self.max_orbitals[i]:
                assert z == Zl[i]  # check that Z is the same for all orbitals
                if l > order_max:
                    order_max = l

        #  error checking
        if self.order < order_max:
            print(
                "An orbital with L={} was found, but the neural network was initialized with L={}".format(
                    order_max, self.order
                )
            )
            print("The neural network MUST have at least the same order as all orbitals!")
            quit()
        if self.order < 2 * order_max:
            print(
                "An orbital with L={} was found, but the neural network was initialized with L={}".format(
                    order_max, self.order
                )
            )
            print(
                "The neural network SHOULD have at least twice the order of the maximum order orbital for good results"
            )
            #  don't quit here, maybe someone wants to do it like this

        #  declare modules and parameters
        self.clebsch_gordan = ClebschGordan()
        self.embedding = SphericalEmbedding(self.order, self.num_features, self.Zmax)
        if self.basis_functions == "exp-gaussian":
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        elif self.basis_functions == "exp-bernstein":
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff
            )
        elif self.basis_functions == "gaussian":
            self.radial_basis_functions = GaussianRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        elif self.basis_functions == "bernstein":
            self.radial_basis_functions = BernsteinRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        else:
            print("basis function type:", self.basis_functions, "is not supported")
        self.module = nn.ModuleList(
            [
                ModularBlock(
                    self.order,
                    self.num_features,
                    self.num_basis_functions,
                    self.num_residual_pre_x,
                    self.num_residual_post_x,
                    self.num_residual_pre_vi,
                    self.num_residual_pre_vj,
                    self.num_residual_post_v,
                    self.num_residual_output,
                    self.clebsch_gordan,
                    True,
                    self.activation,
                )
                for i in range(self.num_modules)
            ]
        )

        self.angular_fn = SphericalLinear(
            self.order,
            1,
            self.order,
            self.num_features,
            clebsch_gordan,
            mix_orders=False,
        )
        self.mix_s = PairMixing(
            self.order,
            self.order,
            self.order,
            self.num_basis_functions,
            self.num_features,
            self.clebsch_gordan,
        )
        self.mix_ij = PairMixing(
            self.order,
            self.order,
            self.order,
            self.num_basis_functions,
            self.num_features,
            self.clebsch_gordan,
        )
        self.radial_ii = nn.ModuleList(
            [nn.Linear(self.num_basis_functions, self.num_features, bias=False) for L in range(self.order + 1)]
        )
        self.radial_ij = nn.ModuleList(
            [nn.Linear(self.num_basis_functions, self.num_features, bias=False) for L in range(self.order + 1)]
        )
        self.residual_pc = ResidualStack(
            self.num_residual_pc,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_pn = ResidualStack(
            self.num_residual_pn,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_ii = ResidualStack(
            self.num_residual_ii,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_ij = ResidualStack(
            self.num_residual_ij,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_full_ii = ResidualStack(
            self.num_residual_full_ii,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_full_ij = ResidualStack(
            self.num_residual_full_ij,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_core_ii = ResidualStack(
            self.num_residual_core_ii,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_core_ij = ResidualStack(
            self.num_residual_core_ij,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )
        self.residual_over_ij = ResidualStack(
            self.num_residual_over_ij,
            self.order,
            self.num_features,
            self.clebsch_gordan,
            True,
            activation,
        )

        if self.activation == "swish":
            self.activation_full_ii = Swish(self.num_features)
            self.activation_full_ij = Swish(self.num_features)
            self.activation_core_ii = Swish(self.num_features)
            self.activation_core_ij = Swish(self.num_features)
            self.activation_over_ij = Swish(self.num_features)
            self.activation_energy = Swish(self.num_energy_features)
        elif activation == "ssp":
            self.activation_full_ii = ShiftedSoftplus(self.num_features)
            self.activation_full_ij = ShiftedSoftplus(self.num_features)
            self.activation_core_ii = ShiftedSoftplus(self.num_features)
            self.activation_core_ij = ShiftedSoftplus(self.num_features)
            self.activation_over_ij = ShiftedSoftplus(self.num_features)
            self.activation_energy = ShiftedSoftplus(self.num_energy_features)
        else:
            print("Unsupported activation function:", activation)
            quit()

        # determine minimum number of output features based on orbitals
        # and generate dictionaries (irreps_ii/irreps_ij) that store indices
        # for collecting the correct irreproducible representations from features
        # diagonal blocks

        number_L = [
            0 for L in range(2 * order_max + 1)
        ]  # keeps track of how many irreps of each order there are already
        self.irreps_ii = {}

        for i in range(len(self.max_orbitals)):
            self.irreps_ii, number_L = self.compute_matrix_irreps(
                self.max_orbitals[i], self.max_orbitals[i], self.irreps_ii, number_L
            )

        self.output_full_ii = SphericalLinear(
            self.order,
            self.num_features,
            2 * order_max,
            max(number_L),
            self.clebsch_gordan,
            zero_init=True,
        )
        self.output_core_ii = SphericalLinear(
            self.order,
            self.num_features,
            2 * order_max,
            max(number_L),
            self.clebsch_gordan,
            zero_init=True,
        )
        self.output_over_ii = SphericalLinear(
            self.order,
            self.num_features,
            2 * order_max,
            max(number_L),
            self.clebsch_gordan,
            zero_init=True,
        )
        for L in range(self.output_over_ii.order_out + 1):  # diagonal blocks of overlap are constant
            self.output_over_ii.linear[L].weight.requires_grad = False  # => only bias terms are used

        # off-diagonal blocks
        number_L = [
            0 for L in range(2 * order_max + 1)
        ]  # keeps track of how many irreps of each order there are already
        self.irreps_ij = {}

        for i in range(len(self.max_orbitals)):
            for j in range(len(self.max_orbitals)):
                if i == j:
                    continue
                self.irreps_ij, number_L = self.compute_matrix_irreps(
                    self.max_orbitals[i], self.max_orbitals[j], self.irreps_ij, number_L
                )

        self.output_full_ij = SphericalLinear(
            self.order,
            self.num_features,
            2 * order_max,
            max(number_L),
            self.clebsch_gordan,
            zero_init=True,
        )
        self.output_core_ij = SphericalLinear(
            self.order,
            self.num_features,
            2 * order_max,
            max(number_L),
            self.clebsch_gordan,
            zero_init=True,
        )
        self.output_over_ij = SphericalLinear(
            self.order,
            self.num_features,
            2 * order_max,
            max(number_L),
            self.clebsch_gordan,
            zero_init=True,
        )

        # initialize parameters
        if load_from is not None:
            try:
                self.load_state_dict(saved_state["model_state_dict"], strict=False)
            except KeyError:
                self.load_state_dict(saved_state["state_dict"], strict=False)
        else:
            self.reset_parameters()

        self.energy_predictor = EnergyLayer(
            self.num_features,
            self.num_energy_features,
            zero_init=False,
            activation=self.activation_energy,
        )

    def reset_parameters(self):
        for L in range(self.order + 1):
            nn.init.orthogonal_(self.radial_ii[L].weight)
            nn.init.orthogonal_(self.radial_ij[L].weight)

    """
    saves the model to a file given by PATH (including all values of the hyperparameters)
    (this file can be passed to the load_from value in the initialization in order to construct
    the model from the saved state)
    """

    def save(self, PATH):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "max_orbitals": self.max_orbitals,
                "order": self.order,
                "num_features": self.num_features,
                "num_basis_functions": self.num_basis_functions,
                "num_modules": self.num_modules,
                "num_residual_pre_x": self.num_residual_pre_x,
                "num_residual_post_x": self.num_residual_post_x,
                "num_residual_pre_vi": self.num_residual_pre_vi,
                "num_residual_pre_vj": self.num_residual_pre_vj,
                "num_residual_post_v": self.num_residual_post_v,
                "num_residual_output": self.num_residual_output,
                "num_residual_pc": self.num_residual_pc,
                "num_residual_pn": self.num_residual_pn,
                "num_residual_ii": self.num_residual_ii,
                "num_residual_ij": self.num_residual_ij,
                "num_residual_full_ii": self.num_residual_full_ii,
                "num_residual_full_ij": self.num_residual_full_ij,
                "num_residual_core_ii": self.num_residual_core_ii,
                "num_residual_core_ij": self.num_residual_core_ij,
                "num_residual_over_ij": self.num_residual_over_ij,
                "basis_functions": self.basis_functions,
                "cutoff": self.cutoff,
                "activation": self.activation,
                "Zmax": self.Zmax,
                "num_energy_features": self.num_energy_features,
            },
            PATH,
        )

    """
    Just for easily printing out the total number of parameters
    """

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    def fill_idx(self, molecule_size, device):
        cum_size = torch.cumsum(molecule_size, dim=0)

        idx_i = torch.cat(
            [
                torch.arange(
                    0 if i == 0 else cum_size[i - 1],
                    cum_size[i],
                    dtype=torch.int64,
                    device=device,
                )
                .view(-1, 1)
                .repeat(1, molecule_size[i])
                .view(-1)
                for i in range(len(molecule_size))
            ]
        )

        idx_j = torch.cat(
            [
                torch.arange(
                    0 if i == 0 else cum_size[i - 1],
                    cum_size[i],
                    dtype=torch.int64,
                    device=device,
                )
                .view(-1, 1)
                .repeat(molecule_size[i], 1)
                .view(-1)
                for i in range(len(molecule_size))
            ]
        )

        self.idx_i, self.idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]

        idx_pi, idx_pj = [], []
        cum_size = 0

        for size in molecule_size:
            curr_size = (size * (size - 1)).item()
            curr_idx_pi, curr_idx_pj = self.idx_pdict[size.item()]
            idx_pi.append(torch.tensor(curr_idx_pi, dtype=torch.int64, device=device) + cum_size)
            idx_pj.append(torch.tensor(curr_idx_pj, dtype=torch.int64, device=device) + cum_size)
            cum_size += curr_size

        self.idx_pi, self.idx_pj = torch.cat(idx_pi), torch.cat(idx_pj)
        self.mask = torch.block_diag(*(torch.ones(size, size) for size in molecule_size))

    """
    Given the Cartesian coordinates and index lists, calculates pairwise distances and unit displacement
    vectors. Each distance/vector is specified by a pair of atom indices i and j (i != j). The total
    number of interactions is num_interactions=num_atoms*(num_atoms-1) when all pairwise distances are
    calculated.

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
        idx_i: indices of atoms i of shape [num_interactions] for collecting Cartesian coordinates
        idx_j: indices of atoms j of shape [num_interactions] for collecting Cartesian coordinates
    outputs:
        dij: pairwise distances of shape [batch_size, num_interactions, 1]
        uij: unit displacement vectors of shape [batch_size, num_interactions, 3]
    """

    @staticmethod
    def calculate_distances_and_directions(R, idx_i, idx_j):
        Ri = torch.gather(
            R,
            -2,
            idx_i.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)),
        )
        Rj = torch.gather(
            R,
            -2,
            idx_j.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)),
        )
        rij = Rj - Ri  # displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
        uij = rij / dij  # unit displacement vectors
        return dij, uij

    """
    Given the orbitals of atom i and atom j, computes how many irreducible representations
    of each order are necessary for constructing the corresponding off-diagonal block of the matrix

    inputs:
        orbitals_i: Tuple or list of tuples with integer entries (Z, L) that define the orbitals of atom i
        orbitals_j: Tuple or list of tuples with integer entries (Z, L) that define the orbitals of atom j
        irreps: Dictionary that stores the feature indices for collecting irreducible representations
        number_L: List of length L+1 with integer entries that stores how many irreducible representations of
                  each order are already in use
    outputs:
        irreps: Updated input dictionary
        number_L: Updated input list
    """

    @staticmethod
    def compute_matrix_irreps(orbitals_i, orbitals_j, irreps, number_L):
        for n_i, orb_i in enumerate(orbitals_i):
            z_i, l_i = orb_i
            for n_j, orb_j in enumerate(orbitals_j):
                z_j, l_j = orb_j
                for L in range(abs(l_i - l_j), l_i + l_j + 1):
                    key = (z_i, z_j, n_i, n_j, L)
                    if key not in irreps.keys():
                        irreps[key] = number_L[L]
                        number_L[L] += 1
        return irreps, number_L

    """
    Given the orbitals in the row and column, constructs a block of the Hamiltonian/Overlap matrix
    from the input irreducible representations

    inputs:
        row: Tuple or list of tuples with integer entries (Z, L) that define the orbitals in the row
        col: Tuple or list of tuples with integer entries (Z, L) that define the orbitals in the column
        irreps: list of irreducible representations of shape [batch_size, 2*L+1]
        batch_size: how many matrices are in this batch (needed to initialize matrix subblock)
    outputs:
        block: batch of matrix blocks of shape [batch_size, nrow, ncol] (nrow/ncol depends on row/col inputs)
    """

    def matrix_block(self, row, col, irreps, batch_size, device="cpu", dtype=torch.float32):
        nrow = sum((2 * l + 1) for z, l in row)  # number of rows in the block
        ncol = sum((2 * l + 1) for z, l in col)  # number of columns in the block
        block = torch.zeros(batch_size, nrow, ncol, device=device, dtype=dtype)

        idx = 0  # index for accessing the correct irreps
        start_i = 0
        for z_i, l_i in row:
            n_i = 2 * l_i + 1
            start_j = 0
            for z_j, l_j in col:
                n_j = 2 * l_j + 1
                for L in range(abs(l_i - l_j), l_i + l_j + 1):
                    # compute inverse spherical tensor product
                    cg = math.sqrt(2 * L + 1) * self.clebsch_gordan(l_i, l_j, L).unsqueeze(0)
                    product = (cg * irreps[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    # add product to appropriate part of the block
                    blockpart = block.narrow(-2, start_i, n_i).narrow(-1, start_j, n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
        return block

    """
    Generate matrix from list of irreps

    inputs:
        irreps: dict with keys (z_i, z_j) and values list of irreps,

    outputs:
        matrix: matrix of shape [batch_size, num_orbitals, num_orbitals]
    """

    def generate_matrix_from_irreps(
        self,
        irreps,
        num_orbitals,
        atom2orbitals,
        begin_index_dict,
        end_index_dict,
        batch_size,
        device,
        dtype,
    ):
        matrix = torch.zeros(batch_size, num_orbitals, num_orbitals, device=device, dtype=dtype)

        for z_i, z_j in irreps:
            current_irreps = irreps[(z_i, z_j)]
            current_irreps_tensors = []
            for L in range(len(current_irreps[0][0])):
                current_irreps_tensors.append(torch.vstack([irrep[L] for irrep, _, _ in current_irreps]))

            blocks = self.matrix_block(
                atom2orbitals[z_i],
                atom2orbitals[z_j],
                current_irreps_tensors,
                batch_size=len(current_irreps),
                device=device,
                dtype=dtype,
            )
            for irreps_id, (irrep, i, j) in enumerate(current_irreps):
                matrix[
                    :,
                    begin_index_dict[i] : end_index_dict[i],
                    begin_index_dict[j] : end_index_dict[j],
                ] = blocks[irreps_id]

        return matrix

    """
    Computes the Hamiltonian/Overlap matrix

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        matrix: Hamiltonian/Overlap matrix of shape [batch_size, num_orbitals, num_orbitals]
    """

    def forward(self, atoms_batch):
        R = atoms_batch["positions"]
        Z = atoms_batch["atomic_numbers"].repeat(1, 1)
        orbitals = atoms_batch["orbitals"]
        atom2orbitals = dict()
        Norb = 0
        begin_index_dict = dict()
        end_index_dict = dict()
        for i in range(len(orbitals)):
            begin_index_dict[i] = Norb
            for z, l in orbitals[i]:
                Norb += 2 * l + 1
            end_index_dict[i] = Norb
            if z not in atom2orbitals:
                atom2orbitals[z] = orbitals[i]

        molecule_size = atoms_batch["molecule_size"]

        self.fill_idx(molecule_size, device=R.device)

        if self.calculate_forces:
            R.requires_grad = True

        # compute radial basis functions and spherical harmonics
        dij, uij = self.calculate_distances_and_directions(R, self.idx_i, self.idx_j)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        sph = spherical_harmonics(self.order, uij)

        for L in range(self.order + 1):
            sph[L].unsqueeze_(-1)  # unsqueeze for broadcasting

        #  initialize atomic features to embeddings
        xs = self.embedding(Z)

        #  the overlap matrix depends only on pairwise information and is therefore
        #  calculated in a different (simpler) branch
        if self.calculate_overlap_matrix:
            fii_over = self.output_over_ii(xs)  # diagonal blocks

            #  construct environment-independent pair features
            si, sj = [], []
            a = self.angular_fn(sph)

            for L in range(self.order + 1):
                i = self.idx_i.view(*(1,) * len(xs[L].shape[:-3]), -1, 1, 1).repeat(
                    *xs[L].shape[:-3], 1, *xs[L].shape[-2:]
                )
                j = self.idx_j.view(*(1,) * len(xs[L].shape[:-3]), -1, 1, 1).repeat(
                    *xs[L].shape[:-3], 1, *xs[L].shape[-2:]
                )
                si.append(torch.gather(xs[L], 1, i))

                if L == 0:
                    sj.append(torch.gather(xs[L], 1, j))
                else:
                    sj.append(a[L])

            sij = self.mix_s(si, sj, rbf)
            fij_over = self.residual_over_ij(sij)
            fij_over[0] = self.activation_over_ij(fij_over[0])
            fij_over = self.output_over_ij(fij_over)

        #  perform iterations over modular building blocks to get environment-dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features

        for module in self.module:
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            for L in range(self.order + 1):
                fs[L] += ys[L]  # add contributions to output features

        fpc = self.residual_pc(fs)  # central pair features
        fpn = self.residual_pn(fs)  # neighbor pair features

        #  compute pair features for self-interactions
        fii = [1 * x for x in fpc]

        for L in range(self.order + 1):  # add influence of neighbouring atoms to pairs
            idx_j = self.idx_j.view(*(1,) * len(fpn[L].shape[:-3]), -1, 1, 1).repeat(
                *fpn[L].shape[:-3], 1, *fpn[L].shape[-2:]
            )
            fpn_j = self.radial_ii[L](rbf) * torch.gather(fpn[L], 1, idx_j)
            fii[L] = fii[L].index_add(1, self.idx_i, fpn_j)

        # compute output features (irreducible representations) for self-interactions
        fii = self.residual_ii(fii)

        if self.calculate_full_hamiltonian:
            fii_full = self.residual_full_ii(fii)
            fii_full[0] = self.activation_full_ii(fii_full[0])
            fii_full = self.output_full_ii(fii_full)

        if self.calculate_core_hamiltonian:
            fii_core = self.residual_core_ii(fii)
            fii_core[0] = self.activation_core_ii(fii_core[0])
            fii_core = self.output_core_ii(fii_core)

        #  gather atomic pairs
        fi = []
        fj = []

        for L in range(self.order + 1):
            i = self.idx_i.view(*(1,) * len(fpc[L].shape[:-3]), -1, 1, 1).repeat(
                *fpc[L].shape[:-3], 1, *fpc[L].shape[-2:]
            )
            j = self.idx_j.view(*(1,) * len(fpc[L].shape[:-3]), -1, 1, 1).repeat(
                *fpc[L].shape[:-3], 1, *fpc[L].shape[-2:]
            )
            fi.append(torch.gather(fpc[L], 1, i))
            fj.append(torch.gather(fpc[L], 1, j))

        #  compute pair features for ordinary interactions
        fij = self.mix_ij(fi, fj, rbf)  # mix pairs

        for L in range(self.order + 1):  # add influence of neighbouring atoms to pairs
            idx_j = self.idx_j.view(*(1,) * len(fpn[L].shape[:-3]), -1, 1, 1).repeat(
                *fpn[L].shape[:-3], 1, *fpn[L].shape[-2:]
            )
            fpn_j = self.radial_ij[L](rbf) * torch.gather(fpn[L], 1, idx_j)
            idx_pj = self.idx_pj.view(*(1,) * len(fpn_j.shape[:-3]), -1, 1, 1).repeat(
                *fpn_j.shape[:-3], 1, *fpn_j.shape[-2:]
            )
            fij[L] = fij[L].index_add(1, self.idx_pi, torch.gather(fpn_j, 1, idx_pj))

        #  compute output features (irreducible representations) for pair-interactions
        fij = self.residual_ij(fij)

        if self.calculate_full_hamiltonian:
            fij_full = self.residual_full_ij(fij)
            fij_full[0] = self.activation_full_ij(fij_full[0])
            fij_full = self.output_full_ij(fij_full)

        if self.calculate_core_hamiltonian:
            fij_core = self.residual_core_ij(fij)
            fij_core[0] = self.activation_core_ij(fij_core[0])
            fij_core = self.output_core_ij(fij_core)

        #  construct batch of matrices of shape [batch_size, num_orbitals, num_orbitals]
        idx = 0  # initialize interaction index to 0 (gets incremented)
        batch_size = fii[0].size(0)
        full_irreps = defaultdict(list)
        core_irreps = defaultdict(list)
        over_irreps = defaultdict(list)
        for i in range(len(orbitals)):  # loop over rows
            for j in range(len(orbitals)):  # loop over columns
                # collect irreps from output features (their shape after squeezing is [batch_size, 2*L+1])
                # features have shape [batch_size,num_atoms/num_interactions,2*L+1,num_features]
                # dimension -3 corresponds to atom/interaction indices
                # dimension -1 corresponds to the feature dimension
                # irreps have shape [batch_size, 2*L+1] (after squeezing)
                full_current_irreps = []
                core_current_irreps = []
                over_current_irreps = []
                if i == j:  # diagonal block
                    for n_i, orb_i in enumerate(orbitals[i]):
                        z_i, l_i = orb_i
                        for n_j, orb_j in enumerate(orbitals[j]):
                            z_j, l_j = orb_j
                            for L in range(abs(l_i - l_j), l_i + l_j + 1):
                                # self.irreps_ii is a dictionary that stores the index ii of the irrep
                                ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
                                if self.calculate_full_hamiltonian:
                                    full_current_irreps.append(
                                        fii_full[L].narrow(-3, i, 1).narrow(-1, ii, 1).squeeze(-3).squeeze(-1)
                                    )
                                if self.calculate_core_hamiltonian:
                                    core_current_irreps.append(
                                        fii_core[L].narrow(-3, i, 1).narrow(-1, ii, 1).squeeze(-3).squeeze(-1)
                                    )
                                if self.calculate_overlap_matrix:
                                    over_current_irreps.append(
                                        fii_over[L].narrow(-3, i, 1).narrow(-1, ii, 1).squeeze(-3).squeeze(-1)
                                    )
                else:  # off-diagonal block
                    if not self.mask[i, j]:
                        continue
                    for n_i, orb_i in enumerate(orbitals[i]):
                        z_i, l_i = orb_i
                        for n_j, orb_j in enumerate(orbitals[j]):
                            z_j, l_j = orb_j
                            for L in range(abs(l_i - l_j), l_i + l_j + 1):
                                # self.irreps_ij is a dictionary that stores the index ij of the irrep
                                ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
                                if self.calculate_full_hamiltonian:
                                    full_current_irreps.append(
                                        fij_full[L].narrow(-3, idx, 1).narrow(-1, ij, 1).squeeze(-3).squeeze(-1)
                                    )
                                if self.calculate_core_hamiltonian:
                                    core_current_irreps.append(
                                        fij_core[L].narrow(-3, idx, 1).narrow(-1, ij, 1).squeeze(-3).squeeze(-1)
                                    )
                                if self.calculate_overlap_matrix:
                                    over_current_irreps.append(
                                        fij_over[L].narrow(-3, idx, 1).narrow(-1, ij, 1).squeeze(-3).squeeze(-1)
                                    )
                    idx += 1  # increment interaction index

                if self.calculate_full_hamiltonian:
                    full_irreps[(z_i, z_j)].append((full_current_irreps, i, j))
                if self.calculate_core_hamiltonian:
                    core_irreps[(z_i, z_j)].append((core_current_irreps, i, j))
                if self.calculate_overlap_matrix:
                    over_irreps[(z_i, z_j)].append((over_current_irreps, i, j))

        #  batch of identity matrices
        eye = torch.eye(Norb, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        if self.calculate_full_hamiltonian:
            full_hamiltonian = self.generate_matrix_from_irreps(
                full_irreps,
                Norb,
                atom2orbitals,
                begin_index_dict,
                end_index_dict,
                batch_size,
                device=R.device,
                dtype=R.dtype,
            )
            full_hamiltonian = full_hamiltonian + full_hamiltonian.transpose(-2, -1)  # symmetrize
        else:
            full_hamiltonian = eye

        if self.calculate_core_hamiltonian:
            core_hamiltonian = self.generate_matrix_from_irreps(
                core_irreps,
                Norb,
                atom2orbitals,
                begin_index_dict,
                end_index_dict,
                batch_size,
                device=R.device,
                dtype=R.dtype,
            )
            core_hamiltonian = core_hamiltonian + core_hamiltonian.transpose(-2, -1)  # symmetrize
        else:
            core_hamiltonian = eye

        if self.calculate_overlap_matrix:
            overlap_matrix = self.generate_matrix_from_irreps(
                over_irreps,
                Norb,
                atom2orbitals,
                begin_index_dict,
                end_index_dict,
                batch_size,
                device=R.device,
                dtype=R.dtype,
            )
            overlap_matrix = overlap_matrix + overlap_matrix.transpose(-2, -1)  # symmetrize
            overlap_matrix = (1 - eye) * overlap_matrix + eye  # set diagonal=1
        else:
            overlap_matrix = eye

        if self.predict_energy:
            pair_sizes = [s * (s - 1) for s in molecule_size]
            energy = self.energy_predictor(fii, fij, list(molecule_size), pair_sizes)

            # the rest is zero
            orbital_energies = torch.zeros_like(torch.diagonal(full_hamiltonian, dim1=-2, dim2=-1))
            orbital_coefficients = torch.zeros_like(full_hamiltonian)
        else:
            orbital_energies = torch.zeros_like(torch.diagonal(full_hamiltonian, dim1=-2, dim2=-1))
            orbital_coefficients = torch.zeros_like(full_hamiltonian)
            energy = torch.zeros((batch_size, 1), device=R.device, dtype=R.dtype, requires_grad=False)

        if self.predict_energy and self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), R, create_graph=self.create_graph)[0]
        else:
            forces = torch.zeros_like(R, requires_grad=self.create_graph)

        results = {
            "full_hamiltonian": full_hamiltonian,
            "core_hamiltonian": core_hamiltonian,
            "overlap_matrix": overlap_matrix,
            "energy": energy,
            "forces": forces,
            "orbital_energies": orbital_energies,
            "orbital_coefficients": orbital_coefficients,
        }
        return results
