# TODO: subject for refactoring
import os
import json
from typing import List
from urllib import request

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Subset
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.data import InMemoryDataset, Data
import nablaDFT


class PyGNablaDFT(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.

    Dataset adapter for ASE2PyG conversion.
    """

    db_suffix = ".db"

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.join(self.datapath, self.dataset_name + self.db_suffix)]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_2k",
        split: str = "train",
        transform=None,
        pre_transform=None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        super(PyGNablaDFT, self).__init__(datapath, transform, pre_transform)

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    def len(self) -> int:
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGNablaDFT, self).get(idx - self.offsets[data_idx])

    def download(self) -> None:
        with open(nablaDFT.__path__[0] + "/links/energy_databases.json", "r") as f:
            data = json.load(f)
            url = data[f"{self.split}_databases"][self.dataset_name]
        request.urlretrieve(url, self.raw_file_names[0])

    def process(self) -> None:
        db = connect(self.raw_file_names[0])
        samples = []
        for db_row in db.select():
            z = torch.from_numpy(db_row.numbers).long()
            positions = torch.from_numpy(db_row.positions).float()
            y = torch.from_numpy(np.array(db_row.data["energy"])).float()
            # TODO: temp workaround for dataset w/o forces
            forces = db_row.data.get("forces", None)
            if forces is not None:
                forces = torch.from_numpy(np.array(forces)).float()
            samples.append(Data(z=z, pos=positions, y=y, forces=forces))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


# From https://github.com/torchmd/torchmd-net/blob/72cdc6f077b2b880540126085c3ed59ba1b6d7e0/torchmdnet/utils.py#L54
def train_val_split(dset_len, train_size, val_size, seed, order=None):
    assert (train_size is None) + (
        val_size is None
    ) <= 1, "Only one of train_size, val_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size

    if train_size is None:
        train_size = dset_len - val_size
    elif val_size is None:
        val_size = dset_len - train_size

    if train_size + val_size > dset_len:
        if is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0, (
        f"One of training ({train_size}), validation ({val_size})"
        f" splits ended up with a negative size."
    )

    total = train_size + val_size

    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        print(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size:total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]

    return np.array(idx_train), np.array(idx_val)


# From: https://github.com/torchmd/torchmd-net/blob/72cdc6f077b2b880540126085c3ed59ba1b6d7e0/torchmdnet/utils.py#L112
def make_splits(
    dataset_len,
    train_size,
    val_size,
    seed,
    filename=None,  # path to save split index
    splits=None,
    order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
    else:
        idx_train, idx_val = train_val_split(
            dataset_len, train_size, val_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
    )


def get_nablaDFT_datasets(
    root: str,
    dataset_name: str,
    train_size: float,
    val_size: float,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    all_dataset = PyGNablaDFT(root, dataset_name)
    idx_train, idx_val = make_splits(
        len(all_dataset),
        train_size,
        val_size,
        seed,
        filename=os.path.join(root, "splits.npz"),
        splits=None,
    )

    train_dataset = Subset(all_dataset, idx_train)
    val_dataset = Subset(all_dataset, idx_val)

    pl_datamodule = LightningDataset(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return pl_datamodule


def get_nablaDFT_test_dataset(
    root: str, dataset_name: str, batch_size: int, num_workers: int
):
    test_dataset = PyGNablaDFT(root, dataset_name, split="test")
    pl_datamodule = LightningDataset(
        train_dataset=None,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return pl_datamodule
