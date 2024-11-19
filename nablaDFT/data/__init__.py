"""Collection of utility classes and functions for chemistry-like datasets."""

import torch_geometric as pyg
from torch.utils.data._utils.collate import default_collate_fn_map

from ._collate import _collate_pyg_batch
from .datasource import Datasource, EnergyDatabase, SQLite3Database
from .hamiltonian_dataset import (
    HamiltonianDatabase,
    HamiltonianDataset,
)
from .nablaDFT_dataset import (  # PyTorch Lightning interfaces for datasets
    ASENablaDFT,
)
from .pl_datamodule import PLDataModule
from .pyg_dataset import PyGDataset  # PyTorch Geometric interfaces for datasets
from .pyg_datasets import (
    PyGHamiltonianNablaDFT,
)
from .registry import dataset_registry  # dataset splits registry

# collate function should be added here
default_collate_fn_map.update({pyg.data.data.BaseData: _collate_pyg_batch})


__all__ = [
    # deprecated
    "HamiltonianDataset",
    "HamiltonianDatabase",
    "ASENablaDFT",
    "PyGNablaDFTDataModule",
    "PyGHamiltonianNablaDFT",
    "dataset_registry",
    # up-to-date
    "EnergyDatabase",
    "SQLite3Database",
    "TorchDataset",
    "PyGDataset",
    "PLDataModule",
    # umbrella type for typing
    "Datasource",
]
