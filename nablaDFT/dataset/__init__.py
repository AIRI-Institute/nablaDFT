"""Collection of utility classes and functions for chemistry-like datasets."""

import torch_geometric as pyg
from torch.utils.data._utils.collate import default_collate_fn_map

from ._collate import collate_pyg
from .hamiltonian_dataset import (  # database interface for Hamiltonian datasets
    HamiltonianDatabase,
    HamiltonianDataset,
)
from .nablaDFT_dataset import (  # PyTorch Lightning interfaces for datasets
    ASENablaDFT,
)
from .pyg_datasets import (  # PyTorch Geometric interfaces for datasets
    PyGHamiltonianNablaDFT,
)
from .registry import dataset_registry  # dataset splits registry

# TODO(aber): collate function should be added here
default_collate_fn_map.update({pyg.data.data.BaseData: collate_pyg})

__all__ = [
    "HamiltonianDataset",
    "HamiltonianDatabase",
    "ASENablaDFT",
    "PyGNablaDFTDataModule",
    "PyGHamiltonianNablaDFT",
    "dataset_registry",
]
