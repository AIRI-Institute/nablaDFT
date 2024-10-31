"""Module for dataset's interfaces using PyTorch Lightning.

Provides functionality for integrating datasets with PyTorch Lightning's DataModule interface.

Examples:
--------
.. code-block:: python
    from nablaDFT.dataset import (
        PyGDataset,
    )

    # Create a new PyGDataset instance
    >>> pyg_dataset = LightningDataModule(
        datasources=datasource,
    )
    >>> pyg_dataset[0]
"""

from typing import List

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData


class PyGDataset(Dataset):
    def __init__(self, datasources: List) -> None:
        super().__init__()
        self.datasources = datasources

    def len(self) -> int:
        pass

    def get(self, idx) -> BaseData:
        pass

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_dir(self) -> str:
        pass

    @property
    def processed_dir(self) -> str:
        pass

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    @property
    def raw_paths(self) -> List[str]:
        pass

    @property
    def processed_paths(self) -> List[str]:
        pass
