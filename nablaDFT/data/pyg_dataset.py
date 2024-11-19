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
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from ._collate import collate_pyg
from ._convert import to_pyg_data
from .utils import check_ds_keys_map, check_ds_len, merge_samples, slice_to_list

logger = logging.getLogger(__name__)


IndexType = Union[int, slice, torch.Tensor, np.ndarray, List]
SampleType = Union[torch.Tensor, np.ndarray]


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
        # prevents processing dataset if in_memory param is False
        return self.in_memory

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
        in_memory: bool = True,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ) -> None:
        if not isinstance(datasources, list):
            datasources = [datasources]
        check_ds_len(datasources)
        check_ds_keys_map(datasources)
        self.datasources = datasources
        self.in_memory = in_memory
        super().__init__(None, transform, pre_transform, pre_filter, False)

        if in_memory:
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
            samples.append(to_pyg_data(data_dict))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]
        data, slices, _ = collate_pyg(samples, increment=False, add_batch=False)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")

    def __getitem__(self, idx: IndexType) -> BaseData:
        """Retrieve elements from dataset."""
        if self.in_memory:
            return super().get(idx)  # just call get from InMemoryDataset
        else:
            if (
                isinstance(idx, (int, np.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))
            ):
                data = {}
                for datasource in self.datasources:
                    data.update(datasource[idx])
                    data = to_pyg_data(data)
                    data = data if self.transform is None else self.transform(data)
                    return data
            else:
                return self.__getitems__(idx)

    def __getitems__(self, idxs: Union[slice, List[int]]) -> List[Dict[str, torch.Tensor]]:
        """Method for multiple samples retrieval.

        Used by pytorch fetcher.
        """
        if isinstance(idxs, slice):
            idxs = slice_to_list(idxs)
        if len(self.datasources) == 1:
            return to_pyg_data(self.datasources[0][idxs])
        else:
            raw_data = [datasource[idxs] for datasource in self.datasources]
            data = to_pyg_data(merge_samples(raw_data))
            data = [sample if self.transform is None else self.transform(sample) for sample in data]
            return data

    def __len__(self):
        """Returns length of dataset."""
        return len(self.datasources[0])
