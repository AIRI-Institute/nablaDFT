"""Module defines Pytorch Lightning DataModule interfaces for various NablaDFT datasets"""
from typing import List, Dict, Optional, Union
import json
import os
from urllib import request as request

from tqdm import tqdm
import numpy as np
from ase.db import connect
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from schnetpack.data import (
    AtomsDataFormat,
    load_dataset,
    AtomsLoader,
    SplittingStrategy,
    AtomsDataModule,
)

import nablaDFT
from nablaDFT.utils import tqdm_download_hook, get_file_size
from .pyg_datasets import PyGNablaDFT, PyGHamiltonianNablaDFT


class ASENablaDFT(AtomsDataModule):
    """PytorchLightning interface for nablaDFT ASE datasets.
    Overrides schnetpack.AtomsDataModule.

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

    Other args description could be found in SchNetPack docs:
    https://schnetpack.readthedocs.io/en/latest/api/generated/data.AtomsDataModule.html#data.AtomsDataModule
    """

    format = AtomsDataFormat.ASE

    def __init__(
        self,
        dataset_name: str,
        split: str,
        datapath: str,
        batch_size: int,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        data_workdir: Optional[str] = None,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = False,
    ):
        """"""
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=self.format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            cleanup_workdir_stage=cleanup_workdir_stage,
            splitting=splitting,
            pin_memory=pin_memory,
        )
        self.split = split
        self._predict_dataset = None
        self._predict_dataloader = None
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
                file_size = get_file_size(url)
                with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, total=file_size, desc=f"Downloading split: {self.dataset_name}") as t:
                    request.urlretrieve(url, self.datapath, reporthook=tqdm_download_hook(t))
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

    def setup(self, stage: Optional[str] = None):
        """Overrides method from original AtomsDataModule class

        Args:
            stage (str): trainer stage, must be one of ['fit', 'test', 'predict']
        """
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()
        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
            )

            # load and generate partitions if needed
            if self.train_idx is None:
                self._load_partitions()

            # partition dataset
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            if self.split == "predict":
                self._predict_dataset = self.dataset.subset(self.test_idx)
            else:
                self._test_dataset = self.dataset.subset(self.test_idx)
            self._setup_transforms()

    def predict_dataloader(self) -> AtomsLoader:
        """Describes predict dataloader, used for prediction task"""
        if self._predict_dataloader is None:
            self._predict_dataloader = AtomsLoader(
                self._predict_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                pin_memory=self._pin_memory,
                shuffle=False,
            )
        return self._predict_dataloader

    def _setup_transforms(self):
        from pdb import set_trace; set_trace()
        for t in self.train_transforms:
            t.datamodule(self)
        for t in self.val_transforms:
            t.datamodule(self)
        for t in self.test_transforms:
            t.datamodule(self)
        self._train_dataset.transforms = self.train_transforms
        self._val_dataset.transforms = self.val_transforms
        if self.split == "test":
            self._test_dataset.transforms = self.test_transforms
        if self.split == "predict":
            self._predict_dataset.transforms = self.test_transforms


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
        return self.dataloader(self.dataset_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return self.dataloader(self.dataset_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return self.dataloader(self.dataset_test, shuffle=False, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return self.dataloader(self.dataset_predict, shuffle=False, **self.dataloader_kwargs)


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
