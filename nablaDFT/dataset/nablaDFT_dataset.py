"""Module defines Pytorch Lightning DataModule interfaces for various NablaDFT datasets"""

import json
import os
from typing import Optional
from urllib import request as request

import numpy as np
import torch
from ase.db import connect
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from schnetpack.data import AtomsDataFormat, load_dataset

import nablaDFT
from .atoms_datamodule import AtomsDataModule
from .pyg_datasets import PyGNablaDFT, PyGHamiltonianNablaDFT


class ASENablaDFT(AtomsDataModule):
    """PytorchLightning interface for nablaDFT ASE datasets.

    Args:
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        datapath (str): path to existing dataset directory or location for download.
        train_ratio (float): dataset part used for training.
        val_ratio (float): dataset part used for validation.
        test_ratio (float): dataset part used for test or prediction.
        train_transforms (Callable): data transform, called for every sample in training dataset.
        val_transforms (Callable): data transform, called for every sample in validation dataset.
        test_transforms (Callable): data transform, called for every sample in test dataset.
    """

    def __init__(
        self,
        split: str,
        dataset_name: str = "dataset_train_2k",
        datapath: str = "database",
        data_workdir: Optional[str] = "logs",
        batch_size: int = 2000,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        **kwargs,
    ):
        """"""
        super().__init__(
            split=split,
            datapath=datapath,
            data_workdir=data_workdir,
            batch_size=batch_size,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            format=format,
            **kwargs,
        )
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def prepare_data(self):
        datapath_with_no_suffix = os.path.splitext(self.datapath)[0]
        suffix = os.path.splitext(self.datapath)[1]
        if not os.path.exists(datapath_with_no_suffix):
            os.makedirs(datapath_with_no_suffix)
        self.datapath = datapath_with_no_suffix + "/" + self.dataset_name + suffix
        exists = os.path.exists(self.datapath)
        if self.split == "predict" and not exists:
            raise FileNotFoundError("Specified dataset not found")
        elif self.split != "predict" and not exists:
            with open(nablaDFT.__path__[0] + "/links/energy_databases.json") as f:
                data = json.load(f)
                if self.train_ratio != 0:
                    url = data["train_databases"][self.dataset_name]
                else:
                    url = data["test_databases"][self.dataset_name]
                request.urlretrieve(url, self.datapath)
        with connect(self.datapath) as ase_db:
            if not ase_db.metadata:
                atomrefs = np.load(
                    nablaDFT.__path__[0] + "/data/atomization_energies.npy"
                )
                ase_db.metadata = {
                    "_distance_unit": "Ang",
                    "_property_unit_dict": {
                        "energy": "Hartree",
                        "forces": "Hartree/Ang",
                    },
                    "atomrefs": {"energy": list(atomrefs)},
                }
            dataset_length = len(ase_db)
            self.num_train = int(dataset_length * self.train_ratio)
            self.num_val = int(dataset_length * self.val_ratio)
            self.num_test = int(dataset_length * self.test_ratio)
            # see AtomsDataModule._load_partitions() for details
            if not self.num_train and not self.num_val:
                self.num_val = -1
                self.num_train = -1
        self.dataset = load_dataset(self.datapath, self.format)


class PyGDataModule(LightningDataModule):
    """Parent class which encapsulates PyG dataset to use with Pytorch Lightning Trainer.
    In order to add new dataset variant, define children class with setup() method.

    Args:
        root (str): path to directory with r'raw/' subfolder with existing dataset or download location.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        train_size (float): part of dataset used for training, must be in [0, 1].
        val_size (float): part of dataset used for validation, must be in [0, 1].
        seed (int): seed number, used for torch.Generator object during train/val split.
        **kwargs: arguments for torch.DataLoader.
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
            "batch_size", "num_workers",
            "pin_memory", "persistent_workers"
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
        return self.dataloader(self.dataset_train, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return self.dataloader(self.dataset_val, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return self.dataloader(self.dataset_test, shuffle=False, **self.kwargs)

    def predict_dataloader(self):
        return self.dataloader(self.dataset_predict, shuffle=False, **self.kwargs)


class PyGHamiltonianDataModule(PyGDataModule):
    """DataModule for Hamiltonian nablaDFT dataset, subclass of PyGDataModule.

    Keyword arguments:
        hamiltonian (bool): retrieve from database molecule's full hamiltonian matrix. True by default.
        include_overlap (bool): retrieve from database molecule's overlab matrix.
        include_core (bool): retrieve from databaes molecule's core hamiltonian matrix.
        **kwargs: arguments for torch.DataLoader and PyGDataModule instance. See PyGDatamodule docs.
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
            dataset = PyGHamiltonianNablaDFT(
                self.root, self.dataset_name, "train", **self.kwargs
            )
            self.dataset_train, self.dataset_val = random_split(
                dataset, self.sizes, generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test":
            self.dataset_test = PyGHamiltonianNablaDFT(
                self.root, self.dataset_name, "test", **self.kwargs
            )
        elif stage == "predict":
            self.dataset_predict = PyGHamiltonianNablaDFT(
                self.root, self.dataset_name, "predict", **self.kwargs
            )


class PyGNablaDFTDataModule(PyGDataModule):
    """DataModule for nablaDFT dataset, subclass of PyGDataModule.
    See PyGDatamodule doc."""

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
            self.dataset_test = PyGNablaDFT(
                self.root, self.dataset_name, "test", **self.kwargs
            )
        elif stage == "predict":
            self.dataset_predict = PyGNablaDFT(
                self.root, self.dataset_name, "predict", **self.kwargs
            )
