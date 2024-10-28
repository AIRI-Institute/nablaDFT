from torch.utils.data._utils.collate import collate_fn_map

from .hamiltonian_dataset import (  # database interface for Hamiltonian datasets
    HamiltonianDatabase,
    HamiltonianDataset,
)
from .nablaDFT_dataset import (  # PyTorch Lightning interfaces for datasets
    ASENablaDFT,
    PyGHamiltonianDataModule,
    PyGNablaDFTDataModule,
)
from .pyg_datasets import (  # PyTorch Geometric interfaces for datasets
    PyGHamiltonianNablaDFT,
    PyGNablaDFT,
)
from .registry import dataset_registry  # dataset splits registry

# TODO: collate function should be added here
collate_fn_map.update({})

__all__ = [
    HamiltonianDataset,
    HamiltonianDatabase,
    ASENablaDFT,
    PyGNablaDFTDataModule,
    PyGHamiltonianDataModule,
    PyGHamiltonianNablaDFT,
    PyGNablaDFT,
    dataset_registry,
]
