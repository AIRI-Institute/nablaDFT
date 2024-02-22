# Overrided AtomsDataModule from SchNetPack
from copy import copy
from typing import List, Dict, Optional, Union

import pytorch_lightning as pl
import torch

from schnetpack.data import (
    AtomsDataFormat,
    load_dataset,
    AtomsLoader,
    SplittingStrategy,
    AtomsDataModule
)


class AtomsDataModule(AtomsDataModule):
    """
    Overrided AtomsDataModule from SchNetPack with additional
    methods for prediction task.
    """

    def __init__(
        self,
        split: str,
        datapath: str,
        batch_size: int,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
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
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = False,
    ):
        """
        Args:
            split: string contains type of task/dataset, must be one of ['train', 'test', 'predict']
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples (absolute or relative)
            num_val: number of validation examples (absolute or relative)
            num_test: number of test examples (absolute or relative)
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then
                batch_size.
            transforms: Preprocessing transform applied to each system separately before
                batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers
                (overrides num_workers).
            num_test_workers: Number of test data loader workers
                (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string
                (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string
                (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. to a local file
                system for faster performance.
            cleanup_workdir_stage: Determines after which stage to remove the data
                workdir
            splitting: Method to generate train/validation/test partitions
                (default: RandomSplit)
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                set to true, when GPUs are used.
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
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
            pin_memory=pin_memory
        )
        self.split = split
        self._predict_dataloader = None
        
    def setup(self, stage: Optional[str] = None):
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
            if self.split == "preidct":
                self._predict_dataset = self.dataset.subset(self.test_idx)
            else:
                self._test_dataset = self.dataset.subset(self.test_idx)
            self._setup_transforms()

    def predict_dataloader(self) -> AtomsLoader:
        if self._predict_dataloader is None:
            self._predict_dataloader = AtomsLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                pin_memory=self._pin_memory,
                shuffle=False
            )
        return self._predict_dataloader
