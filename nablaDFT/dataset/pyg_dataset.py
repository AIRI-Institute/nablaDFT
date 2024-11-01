"""Module for dataset's interfaces using PyTorch Geometric.

Provides functionality for integrating datasources with PyTorch Geometric Datasets.

Examples:
--------
.. code-block:: python
    from nablaDFT.dataset import (
        PyGDataset,
    )

    # Create a new PyGDataset instance from datasource
    >>> pass
    >>> pass
"""

from typing import List

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData

from .utils import _check_ds_len


class PyGDataset(Dataset):
    def __init__(self, datasources: List) -> None:
        _check_ds_len(datasources)
        super().__init__()
        if isinstance(datasources, str):
            datasources = [datasources]
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
