import torch_geometric as pyg
from torch.utils.data._utils.collate import default_collate_fn_map

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
default_collate_fn_map.update(
    {pyg.data.data.BaseData: pyg.data.batch.Batch.from_data_list}
)  # or use pyg.data.collate()?

__all__ = [
    "HamiltonianDataset",
    "HamiltonianDatabase",
    "ASENablaDFT",
    "PyGNablaDFTDataModule",
    "PyGHamiltonianNablaDFT",
    "dataset_registry",
]
