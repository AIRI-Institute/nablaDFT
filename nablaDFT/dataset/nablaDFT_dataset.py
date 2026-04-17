"""Module defines Pytorch Lightning DataModule interfaces for nablaDFT datasets"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from ase.db import connect
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import nablaDFT
from nablaDFT.dataset.registry import dataset_registry
from nablaDFT.utils import download_file

from .pyg_datasets import PyGHamiltonianNablaDFT, PyGNablaDFT


class PyGDataModule(LightningDataModule):
    """Base class which encapsulates PyG dataset for usage with Pytorch Lightning Trainer.

    Args:
        root (str): path to directory with :obj: raw/ subfolder with existing dataset or download location.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        train_size (float): part of dataset used for training, must be in [0, 1].
        val_size (float): part of dataset used for validation, must be in [0, 1].
        .. note::
            :obj: train_size and :obj: val_size are not used during :obj: test or :obj: predict pipelines.
        seed (int): seed number, used for torch.Generator object during train/val split.
        **kwargs: additional arguments for torch.DataLoader.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        train_size: float = 0.9,
        val_size: float = 0.1,
        seed: int = 23,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_predict = None

        self.root = root
        self.dataset_name = dataset_name
        self.seed = seed
        self.sizes = [train_size, val_size]
        dataloader_keys = [
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
        ]
        self.dataloader_kwargs = {}
        for key in dataloader_keys:
            val = kwargs.get(key, None)
            self.dataloader_kwargs[key] = val
            if val is not None:
                del kwargs[key]
        self.kwargs = kwargs

    def dataloader(self, dataset, **kwargs):
        return DataLoader(dataset, **kwargs)

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return self.dataloader(self.dataset_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return self.dataloader(self.dataset_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return self.dataloader(self.dataset_test, shuffle=False, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return self.dataloader(self.dataset_predict, shuffle=False, **self.dataloader_kwargs)


class PyGHamiltonianDataModule(PyGDataModule):
    """DataModule for Hamiltonian nablaDFT dataset.

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/hamiltonian_databases.json), dataset will be downloaded automatically.

    Args:
        include_hamiltonian (bool): retrieve from database molecule's full hamiltonian matrix. Default is :obj: True.
        include_overlap (bool): retrieve from database molecule's overlab matrix.
        include_core (bool): retrieve from database molecule's core hamiltonian matrix.

    See :class:`nablaDFT.dataset.nablaDFT_dataset.PyGDataModule` for other parameters' description.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        train_size: float = None,
        val_size: float = None,
        include_hamiltonian: bool = True,
        include_overlap: bool = False,
        include_core: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            root,
            dataset_name,
            train_size,
            val_size,
            include_hamiltonian=include_hamiltonian,
            include_overlap=include_overlap,
            include_core=include_core,
            **kwargs,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = PyGHamiltonianNablaDFT(self.root, self.dataset_name, "train", **self.kwargs)
            self.dataset_train, self.dataset_val = random_split(
                dataset, self.sizes, generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test":
            self.dataset_test = PyGHamiltonianNablaDFT(self.root, self.dataset_name, "test", **self.kwargs)
        elif stage == "predict":
            self.dataset_predict = PyGHamiltonianNablaDFT(self.root, self.dataset_name, "predict", **self.kwargs)


class PyGNablaDFTDataModule(PyGDataModule):
    """DataModule for nablaDFT dataset, subclass of PyGDataModule.

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/energy_databases.json), dataset will be downloaded automatically.

    See :class:`nablaDFT.dataset.nablaDFT_dataset.PyGDataModule` for reference.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        train_size: float = None,
        val_size: float = None,
        **kwargs,
    ) -> None:
        super().__init__(root, dataset_name, train_size, val_size, **kwargs)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = PyGNablaDFT(self.root, self.dataset_name, "train", **self.kwargs)
            self.dataset_train, self.dataset_val = random_split(
                dataset, self.sizes, generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test":
            self.dataset_test = PyGNablaDFT(self.root, self.dataset_name, "test", **self.kwargs)
        elif stage == "predict":
            self.dataset_predict = PyGNablaDFT(self.root, self.dataset_name, "predict", **self.kwargs)

