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

import logging
from typing import Callable, List, Union

import torch
import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData, Data

from .utils import _check_ds_len

logger = logging.getLogger(__name__)


class PyGDataset(InMemoryDataset):
    """Pytorch Geometric interface for datasets.

    Stores elements in memory, for more information check PyG `docs
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.InMemoryDataset.html#torch_geometric.data.InMemoryDataset>`__

    .. code-block:: python
        from nablaDFT.dataset import PyGDataset

        dataset = PyGNablaDFT(datasource)
        sample = dataset[0]

    Args:
        datasources (str): path to existing dataset directory or location for download.
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    def __init__(
        self,
        datasources: Union[str, List[str]],
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ) -> None:
        _check_ds_len(datasources)
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        if isinstance(datasources, str):
            datasources = [datasources]
        self.datasources = datasources
        super().__init__(None, transform, pre_transform, pre_filter, False)

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def get(self, idx) -> BaseData:
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGDataset, self).get(idx - self.offsets[data_idx])

    def process(self):
        samples = []
        for idx in tqdm(range(len(self.datasources[0])), total=len(self.datasources[0])):
            data_dict = {}
            for datasource in self.datasources:
                data_dict.update(datasource[idx])
            samples.append(Data(**data_dict))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")

    @property
    def raw_dir(self) -> str:
        # TODO: need to overwrite this to prevent `/raw` subfolder creation
        return self.raw_dir

    @property
    def processed_dir(self) -> str:
        # TODO: need to overwrite this to prevent `/processed` subfolder creation
        return self.processed_dir

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass
