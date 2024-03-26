"""Module describes PyTorch Geometric interfaces for various NablaDFT datasets"""
import json
import os
import logging
from typing import List, Callable
from urllib import request as request

from tqdm import tqdm
import numpy as np
import torch
from ase.db import connect
from torch_geometric.data import InMemoryDataset, Data, Dataset

import nablaDFT
from .hamiltonian_dataset import HamiltonianDatabase
from nablaDFT.utils import tqdm_download_hook, get_file_size

logger = logging.getLogger(__name__)


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
        transform: Callable = None,
        pre_transform: Callable = None,
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
        file_size = get_file_size(url)
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, total=file_size, desc=f"Downloading split: {self.dataset_name}") as t:
            request.urlretrieve(url, self.raw_paths[0], reporthook=tqdm_download_hook(t))

    def process(self) -> None:
        db = connect(self.raw_paths[0])
        samples = []
        for db_row in tqdm(db.select(), total=len(db)):
            z = torch.from_numpy(db_row.numbers).long()
            positions = torch.from_numpy(db_row.positions).float()
            y = torch.from_numpy(np.array(db_row.data["energy"])).float()
            forces = torch.from_numpy(np.array(db_row.data["forces"])).float()
            samples.append(Data(z=z, pos=positions, y=y, forces=forces))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")


class PyGHamiltonianNablaDFT(Dataset):
    """Pytorch Geometric dataset for NablaDFT Hamiltonian database.

    Args:
      - datapath (str): path to existing dataset directory or location for download.
      - dataset_name (str): split name from links .json.
      - split (str): type of split, must be one of ['train', 'test', 'predict'].
      - include_hamiltonian (bool): if True, retrieves full Hamiltonian matrices from database.
      - include_overlap (bool): if True, retrieves overlap matrices from database.
      - include_core (bool): if True, retrieves core Hamiltonian matrices from database.
      - dtype (torch.dtype): defines torch.dtype for energy, positions, forces tensors.
      - transform (Callable): callable data transform, called on every access to element.
      - pre_transform (Callable): callable data transform, called on every access to element.
    Note:
        Hamiltonian matrix for each molecule has different shape. PyTorch Geometric tries to concatenate
        each torch.Tensor in batch, so in order to make batch from data we leave all hamiltonian matrices
        in numpy array form. During train, these matrices will be yield as List[np.array].
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
        datapath="database",
        dataset_name="dataset_train_2k",
        split: str="train",
        include_hamiltonian: bool = True,
        include_overlap: bool = False,
        include_core: bool = False,
        dtype=torch.float32,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        self.dtype = dtype
        self.include_hamiltonian = include_hamiltonian
        self.include_overlap = include_overlap
        self.include_core = include_core

        super(PyGHamiltonianNablaDFT, self).__init__(datapath, transform, pre_transform)

        self.max_orbitals = self._get_max_orbitals(datapath, dataset_name)
        self.db = HamiltonianDatabase(self.raw_paths[0])

    def len(self) -> int:
        return len(self.db)

    def get(self, idx):
        data = self.db[idx]
        z = torch.tensor(data[0]).long()
        positions = torch.tensor(data[1]).to(self.dtype)
        # see notes
        hamiltonian = data[4]
        if self.include_overlap:
            overlap = data[5]
        else:
            overlap = None
        if self.include_core:
            core = data[6]
        else:
            core = None
        y = torch.from_numpy(data[2]).to(self.dtype)
        forces = torch.from_numpy(data[3]).to(self.dtype)
        data = Data(
            z=z, pos=positions,
            y=y, forces=forces,
            hamiltonian=hamiltonian,
            overlap=overlap,
            core=core,
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    def download(self) -> None:
        with open(nablaDFT.__path__[0] + "/links/hamiltonian_databases.json") as f:
                data = json.load(f)
                url = data["train_databases"][self.dataset_name]
        file_size = get_file_size(url)
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, total=file_size, desc=f"Downloading split: {self.dataset_name}") as t:
            request.urlretrieve(url, self.raw_paths[0], reporthook=tqdm_download_hook(t))

    def process(self) -> None:
        pass

    def _get_max_orbitals(self, datapath, dataset_name):
        db_path = os.path.join(datapath, "raw/" + dataset_name + self.db_suffix)
        if not os.path.exists(db_path):
            self.download()
        database = HamiltonianDatabase(db_path)
        max_orbitals = []
        for z in database.Z:
            max_orbitals.append(
                tuple((int(z), int(l)) for l in database.get_orbitals(z))
            )
        max_orbitals = tuple(max_orbitals)
        return max_orbitals
