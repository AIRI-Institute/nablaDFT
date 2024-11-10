"""Module for dataset's interfaces using PyTorch Geometric.

Provides functionality for integrating datasources with PyTorch Geometric Datasets.

Examples:
--------
.. code-block:: python
    from nablaDFT.dataset import (
        PyGDataset,
    )

    >>> datasource = EnergyDatabase("path-to-database")
    # Create a new PyGDataset instance from datasource
    >>> dataset = PyGDataset(datasource)
"""

import logging
import pathlib
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_convert
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData, Data
from tqdm import tqdm

from ._collate import collate_pyg
from .utils import _check_ds_len

logger = logging.getLogger(__name__)


IndexType = Union[slice, Tensor, np.ndarray, List]


class PyGDataset(InMemoryDataset):
    """Pytorch Geometric interface for datasets.

    Stores elements in memory, for more information check PyG `docs
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.InMemoryDataset.html#torch_geometric.data.InMemoryDataset>`__

    .. code-block:: python
        from nablaDFT.dataset import PyGDataset

        >>> dataset = PyGNablaDFT(datasource)
        >>> sample = dataset[0]
        >>> Data(y=[...], pos[..., 3], forces[..., 3])

    .. note:: datasources must be in the same directory.
    .. warn:: element index must belong to the same conformations/molecule in different data sources.

    Args:
        datasources (str): path to existing dataset directory or location for download.
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    @property
    def has_process(self):
        # prevent processing dataset if in_memory param is False or None
        if self.in_memory:
            return True
        return False

    @property
    def raw_file_names(self) -> List[str]:
        return [datasource.filepath.name for datasource in self.datasources]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{datasource.filepath.stem}_processed.pt" for datasource in self.datasources]

    @property
    def raw_dir(self) -> List[pathlib.Path]:
        return self.datasources[0].filepath.parent

    @property
    def processed_dir(self) -> List[pathlib.Path]:
        return self.raw_dir / pathlib.Path("processed")

    def __init__(
        self,
        datasources: Union[str, List[str]],
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ) -> None:
        if not isinstance(datasources, list):
            datasources = [datasources]
        _check_ds_len(datasources)
        self.datasources = datasources
        super().__init__(None, transform, pre_transform, pre_filter, False)

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data = data
            self.slices = slices

    def process(self):
        samples = []
        for idx in tqdm(range(len(self.datasources[0])), total=len(self.datasources[0])):
            data_dict = {}
            for datasource in self.datasources:
                data_dict.update(datasource[idx])
            samples.append(self._data_from_sample(data_dict))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]
        data, slices, _ = collate_pyg(samples, increment=False, add_batch=False)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")

    def __getitem__(self, idx) -> BaseData:
        # TODO: Do we need to appopriate describe case of several indices for in_memory=False???
        if self.in_memory:
            super().get(idx)
        else:
            if (
                isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))
            ):
                data = {}
                for datasource in self.datasources:
                    data.update(datasource[idx])
                    data = self._data_from_sample(data)
                    data = data if self.transform is None else self.transform(data)
                    return data

    def _data_from_sample(self, sample: Dict[str, Union[np.ndarray, torch.Tensor]]) -> BaseData:
        for key in sample.keys():
            sample[key] = default_convert(sample[key])
        return Data(**sample)


"""
    def get(self, idx) -> BaseData:
        data_idx = 0
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGDataset, self).get(idx - self.offsets[data_idx])
"""
