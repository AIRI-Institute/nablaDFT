"""Module for dataset's interfaces using PyTorch.

Provides functionality for integrating datasets with PyTorch.

Examples:
--------
.. code-block:: python
    from nablaDFT.dataset import (
        TorchDataset,
    )

    >>> pass
    >>> pass
"""

from typing import Dict, List, Mapping, Union

import torch
from torch.utils.data import Dataset

from .utils import _check_ds_len


class TorchDataset(Dataset):
    """Dataset interface for PyTorch.

    Combines one or more DataSource objects into one dataset.

    ..note: size of datasource elements must be the same.

    Args:
        datasources (Union[Mapping, List[Mapping]]): datasources objects.
    """

    def __init__(self, datasources: Union[Mapping, List[Mapping]]) -> None:
        _check_ds_len(datasources)
        super().__init__()
        if isinstance(datasources, str):
            datasources = [datasources]
        self.datasources = datasources

    def __geitem__(self, idx: int) -> Dict[torch.Tensor]:
        """Return single dataset element.

        Args:
            idx (int): element index.

        Returns:
            data (Doct[torch.Tensor]): dataset element.
        """
        data = {}
        for datasource in self.datasources:
            data.update(datasource[idx])
        return data

    def __getitems__(self, idx: Union[List[int], slice]) -> Dict[torch.Tensor]:
        """Returns multiple dataset elements.

        Datasources must support simultaneous access.

        Args:
            idx (List[int]): indexes to get.

        Returns:
            Dict[torch.Tensor]: data
        """
        data = {}
        for datasource in self.datasources:
            data.update(datasource[idx])
        return data

    def __len__(self) -> int:
        """Returns length of dataset."""
        return len(self.datasources[0])
