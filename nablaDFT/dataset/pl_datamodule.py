"""
Module for working with nablaDFT datasets using PyTorch Lightning.

Provides functionality for integrating datasets with PyTorch Lightning's DataModule interface.

Examples
--------
.. code-block:: python
    from nablaDFT.dataset import (
        LightningDataModule,
    )

    # Create a new LightningDataModule instance
    datamodule = LightningDataModule(
        dataset=my_dataset,
        train_size=0.9,
        num_workers=9,
        batch_size=32,
    )

    # Pass datamodule to pl.Trainer
    trainer.fit(
        model=model, datamodule=datamodule
    )
"""

from typing import Optional

import torch
from pytorch_lightning import LightningDataModule as PLDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split


class LightningDataModule(PLDataModule):
    """
    Wraps dataset for PyTorchLightning DataModule interface.

    Attributes:
        dataset (Dataset): The wrapped dataset.
        train_size (Optional[float]): The proportion of the dataset used for training, used only in fit stage.
        kwargs: arguments for :class: torch.utils.data.DataLoader, see `DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__.
    """

    def __init__(
        self,
        dataset: Dataset,
        train_size: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.train_size = train_size

        # Define defaults for DataLoader
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("num_workers", 0)
        kwargs.setdefault("pin_memory", True)

        shuffle = isinstance(dataset, IterableDataset) | kwargs.get("sampler", None) | kwargs.get("batch_sampler", None)
        kwargs.setdefault("shuffle", shuffle)
        kwargs.setdefault("persistent_workers", kwargs.get("num_workers", 0) > 0)

        self.kwargs = kwargs

    def dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.train_size is None:
                raise ValueError("train_size and val_size must be set for fit stage")
            val_size = 1 - self.train_size
            sizes = [int(self.train_size * len(self.dataset)), int(val_size * len(self.dataset))]
            self.dataset_train, self.dataset_val = random_split(dataset, sizes)
        elif stage == "test":
            self.dataset_test = dataset
        elif stage == "predict":
            self.dataset_predict = dataset

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.dataset_train, **self.kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs["shuffle"] = False
        return self.dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs["shuffle"] = False
        return self.dataloader(self.dataset_val, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        kwargs = self.kwargs.copy()
        kwargs["shuffle"] = False
        return self.dataloader(self.dataset_val, **kwargs)