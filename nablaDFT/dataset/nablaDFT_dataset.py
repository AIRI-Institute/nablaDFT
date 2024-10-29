"""Module defines Pytorch Lightning DataModule interfaces for nablaDFT datasets"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from ase.db import connect
from pytorch_lightning import LightningDataModule
from schnetpack.data import (
    AtomsDataFormat,
    AtomsDataModule,
    AtomsLoader,
    BaseAtomsData,
    SplittingStrategy,
    load_dataset,
)
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import nablaDFT
from nablaDFT.dataset.registry import dataset_registry
from nablaDFT.utils import download_file

from .pyg_datasets import PyGHamiltonianNablaDFT, PyGNablaDFT


class ASENablaDFT(AtomsDataModule):
    """PytorchLightning interface for nablaDFT datasets for SchNetPack based models.

    Inherits from `schnetpack.data.AtomsDataModule <https://schnetpack.readthedocs.io/en/latest/api/generated/data.AtomsDataModule.html#data.AtomsDataModule>`_.

    Args:
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        root (str): path to existing dataset directory or location for download.
        train_ratio (float): dataset part used for training.
        val_ratio (float): dataset part used for validation.
        test_ratio (float): dataset part used for test or prediction.
        train_transforms (Callable): data transform, called for every sample in training dataset.
        val_transforms (Callable): data transform, called for every sample in validation dataset.
        test_transforms (Callable): data transform, called for every sample in test dataset.
    """

    format = AtomsDataFormat.ASE

    def __init__(
        self,
        dataset_name: str,
        split: str,
        root: str,
        batch_size: int,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        data_workdir: Optional[str] = None,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = None,
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
        seed: int = 47,
    ):
        if not Path(root).exists():
            Path(root).mkdir(parents=True, exist_ok=True)
        super().__init__(
            datapath=root,
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
        self._predict_dataset = None
        self._predict_dataloader = None
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split = split
        self.seed = seed

        self._resolve_path(root)

    def prepare_data(self) -> None:
        if not self.datapath.exists():
            self._download()

    def setup(self, stage: Optional[str] = None):
        """Overrides method from original AtomsDataModule class

        Args:
            stage (str): trainer stage, must be one of ['fit', 'test', 'predict']
        """
        # check whether data needs to be copied
        # (re)load datasets
        with connect(self.datapath) as ase_db:
            self._check_metadata(ase_db)
            dataset_length = len(ase_db)
            self.num_train = int(dataset_length * self.train_ratio)
            self.num_val = int(dataset_length * self.val_ratio)
            self.num_test = int(dataset_length * self.test_ratio)
            # see AtomsDataModule._load_partitions() for details
            if not self.num_train and not self.num_val:
                self.num_val = -1
                self.num_train = -1
        self.dataset = load_dataset(
            self.datapath,
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
            self._test_dataset = self.dataset.subset([])
        else:
            self._test_dataset = self.dataset.subset(self.test_idx)
            self._predict_dataset = self.dataset.subset([])
        self._setup_transforms()

    @property
    def predict_dataset(self) -> BaseAtomsData:
        return self._predict_dataset

    def predict_dataloader(self) -> AtomsLoader:
        """Returns predict dataloader, used for prediction task"""
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

    def _download(self):
        url = dataset_registry.get_dataset_url("energy", self.dataset_name)
        dataset_etag = dataset_registry.get_dataset_etag("energy", self.dataset_name)
        download_file(
            url,
            Path(self.datapath),
            dataset_etag,
            desc=f"Downloading split: {self.dataset_name}",
        )

    def _check_metadata(self, conn):
        if not conn.metadata:
            atomrefs = np.load(nablaDFT.__path__[0] + "/data/atomization_energies.npy")
            conn.metadata = {
                "_distance_unit": "Ang",
                "_property_unit_dict": {
                    "energy": "Hartree",
                    "forces": "Hartree/Ang",
                },
                "atomrefs": {"energy": list(atomrefs)},
            }

    def _resolve_path(self, root):
        datapath_dir = Path(root).resolve()
        if not datapath_dir.is_dir():
            datapath_dir = Path(".").resolve()
        # suffix is always '.db', because other formats are not supported by schnetpack
        suffix = "db"
        datapath_dir.mkdir(parents=True, exist_ok=True)
        self.datapath = datapath_dir / f"{self.dataset_name}.{suffix}"
