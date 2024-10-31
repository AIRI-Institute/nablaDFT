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

from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, datasources: Union[int, str]) -> None:
        pass

    def __geitem__(self, idx) -> Dict[torch.Tensor]:
        pass

    def __getitems__(self, idx) -> Dict[torch.Tensor]:
        pass

    def __len__(self) -> int:
        pass
