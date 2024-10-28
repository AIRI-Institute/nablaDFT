"""Module defines Pytorch Lightning DataModule interfaces for nablaDFT datasets"""

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class DataModule(LightningDataModule):
    """Module defines Pytorch Lightning DataModule interfaces for nablaDFT datasets"""

    def __init__(
        self,
        dataset: Dataset,  # already created dataset
        dataloader_cls: DataLoader,  # or just pass raw interface (pytorch or pyg) to create dataloader?
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset = dataset  # do we need here some weakref.proxy?
        self.dataloader_cls = dataloader_cls
        self.krwargs = kwargs

    def dataloader(self, dataset, **kwargs) -> DataLoader:
        return self.dataloader_cls(dataset, **kwargs)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.dataset_train, self.dataset_val = random_split(
                dataset, self.sizes, generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test":
            self.dataset_test = dataset
        elif stage == "predict":
            self.dataset_predict = dataset

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.dataset_train, shuffle=True, **self.kwargs)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.dataset_val, shuffle=False, **self.kwargs)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.dataset_test, shuffle=False, **self.kwargs)

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(self.dataset_predict, shuffle=False, **self.kwargs)
