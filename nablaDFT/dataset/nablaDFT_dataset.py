import json
import os
from typing import Optional, List
from urllib import request as request

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Subset
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.data import InMemoryDataset, Data
from schnetpack.data import AtomsDataFormat, AtomsDataModule, load_dataset
import nablaDFT


from nablaDFT.phisnet.training.hamiltonian_dataset import HamiltonianDataset
from nablaDFT.phisnet.training.sqlite_database import HamiltonianDatabase


class ASENablaDFT(AtomsDataModule):
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
            with open(nablaDFT.__path__[0] + "/links/energy_databases_v2.json") as f:
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


class HamiltonianNablaDFT(HamiltonianDataset):
    def __init__(
        self,
        datapath="database",
        dataset_name="dataset_train_2k",
        max_batch_orbitals=1200,
        max_batch_atoms=150,
        max_squares=4802,
        subset=None,
        dtype=torch.float32,
    ):
        self.dtype = dtype
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        f = open(nablaDFT.__path__[0] + "/links/hamiltonian_databases.json")
        data = json.load(f)
        url = data["train_databases"][dataset_name]
        f.close()
        filepath = datapath + "/" + dataset_name + ".db"
        request.urlretrieve(url, filepath)
        self.database = HamiltonianDatabase(filepath)
        max_orbitals = []
        for z in self.database.Z:
            max_orbitals.append(
                tuple((int(z), int(l)) for l in self.database.get_orbitals(z))
            )
        max_orbitals = tuple(max_orbitals)
        self.max_orbitals = max_orbitals
        self.max_batch_orbitals = max_batch_orbitals
        self.max_batch_atoms = max_batch_atoms
        self.max_squares = max_squares
        self.subset = None
        if subset:
            self.subset = np.load(subset)


class PyGNablaDFT(InMemoryDataset):
    """Dataset adapter for ASE2PyG conversion.
    Based on https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/md17.py
    """

    db_suffix = ".db"

    @property
    def raw_file_names(self) -> List[str]:
        return [(self.dataset_name + self.db_suffix)]

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
        with open(nablaDFT.__path__[0] + "/links/energy_databases_v2.json", "r") as f:
            data = json.load(f)
            url = data[f"{self.split}_databases"][self.dataset_name]
        request.urlretrieve(url, self.raw_paths[0])

    def process(self) -> None:
        db = connect(self.raw_paths[0])
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


def get_PyG_nablaDFT_datasets(
    root: str,
    split: str,
    dataset_name: str,
    train_size: float = None,
    val_size: float = None,
    batch_size: int = None,
    num_workers: int = None,
    seed: int = None,
):
    dataset = PyGNablaDFT(root, dataset_name, split=split)
    if split == "train":
        idx_train, idx_val = make_splits(
            len(dataset),
            train_size,
            val_size,
            seed,
            filename=os.path.join(root, "splits.npz"),
            splits=None,
        )
        train_dataset = Subset(dataset, idx_train)
        val_dataset = Subset(dataset, idx_val)
        test_dataset = None
    else:
        train_dataset = None
        val_dataset = None
        test_dataset = dataset

    pl_datamodule = LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return pl_datamodule


class NablaDFT:
    def __init__(self, type_of_nn, *args, **kwargs):
        valid = {"ASE", "Hamiltonian", "PyG"}
        if type_of_nn not in valid:
            raise ValueError("results: type of nn must be one of %r." % valid)
        self.type_of_nn = type_of_nn
        if self.type_of_nn == "ASE":
            self.dataset = ASENablaDFT(*args, **kwargs)
        elif self.type_of_nn == "Hamiltonian":
            self.dataset = HamiltonianNablaDFT(*args, **kwargs)
        else:
            self.dataset = get_PyG_nablaDFT_datasets(*args, **kwargs)
