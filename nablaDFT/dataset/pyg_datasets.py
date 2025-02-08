"""Module describes PyTorch Geometric interfaces for nablaDFT datasets"""

import logging
import os
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from ase.db import connect
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm
import lmdb
import pickle
import gzip
import shutil
import tarfile
import pandas as pd
from multiprocessing import Pool

import json

try:
    from rdkit import Chem
except:
    pass

from nablaDFT.dataset.registry import dataset_registry
from nablaDFT.utils import (
    download_file, 
    decide_download, 
    download_url,
    get_atomic_number,
    mol2graph,
    extract_zip, 
    replace_numpy_with_torchtensor
)

from .hamiltonian_dataset import HamiltonianDatabase

logger = logging.getLogger(__name__)


class PyGNablaDFT(InMemoryDataset):
    """Pytorch Geometric interface for nablaDFT datasets.

    Based on `MD17 implementation <https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/md17.py>`_.

    .. code-block:: python
        from nablaDFT.dataset import PyGNablaDFT

        dataset = PyGNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/energy_databases.json), dataset will be downloaded automatically.

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
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
        dataset_name: str = "dataset_train_tiny",
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
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self) -> int:
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGNablaDFT, self).get(idx - self.offsets[data_idx])

    def download(self) -> None:
        url = dataset_registry.get_dataset_url("energy", self.dataset_name)
        dataset_etag = dataset_registry.get_dataset_etag("energy", self.dataset_name)
        download_file(
            url,
            Path(self.raw_paths[0]),
            dataset_etag,
            desc=f"Downloading split: {self.dataset_name}",
        )

    def process(self) -> None:
        db = connect(self.raw_paths[0])
        samples = []
        for db_row in tqdm(db.select(), total=len(db)):
            z = torch.from_numpy(db_row.numbers.copy()).long()
            positions = torch.from_numpy(db_row.positions.copy()).float()
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


class PyGFluoroDataset(InMemoryDataset):
    """Pytorch Geometric interface for nablaDFT datasets.

    Based on `MD17 implementation <https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/md17.py>`_.

    .. code-block:: python
        from nablaDFT.dataset import PyGNablaDFT

        dataset = PyGNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/energy_databases.json), dataset will be downloaded automatically.

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    db_suffix = ".lmdb"

    @property
    def raw_file_names(self) -> List[str]:
        return [(f"{self.split}_{self.dataset_name}{self.db_suffix}")]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        print(self.raw_file_names)
        super(PyGFluoroDataset, self).__init__(datapath, transform, pre_transform)

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self) -> int:
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGFluoroDataset, self).get(idx - self.offsets[data_idx])

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        env_label_3D = lmdb.open(
            self.raw_paths[0],
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )

        with env_label_3D.begin() as txn:
            length = txn.stat()['entries']
            samples = []
            for key, value in tqdm(txn.cursor(), total=length):
                dt = pickle.loads(gzip.decompress(value))
                z = np.array([get_atomic_number(elem) for elem in dt['atoms']])
                if len(z) == 1:
                    print (key, dt['smi'])
                    continue
                z = torch.from_numpy(z).long()
                y = torch.from_numpy(np.array([dt['target']])).float()
                #for positions in dt['input_pos']:
                #    pos = torch.from_numpy(positions).float()
                #    samples.append(Data(z=z, pos=pos, y=y))
                positions = dt['label_pos']
                pos = torch.from_numpy(positions).float()
                samples.append(Data(z=z, pos=pos, y=y))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        transform=None,
        pre_transform=None,
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m-v2
           
        refer to: https://github.com/lsj2408/Transformer-M/blob/main/Transformer-M/data/wrapper.py
        """

        self.original_root = root
        self.folder = os.path.join(root, "pcqm4m-v2")
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = (
            "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        )
        self.pos_url = (
            "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz"
        )

        # check version and update if necessary
        if os.path.isdir(self.folder) and (
            not os.path.exists(os.path.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2PosDataset, self).__init__(
            self.folder, transform, pre_transform
        )

        fn_processed = self.processed_paths[0]
        print(f"Loading data from {fn_processed}")
        self.data, self.slices = torch.load(fn_processed)

    @property
    def raw_file_names(self):
        return ["data.csv.gz", "pcqm4m-v2-train.sdf"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed_3dm_v2.pt"

    def download(self):
        if int(os.environ.get("RANK", 0)) == 0:
            # This is for multiple GPUs in one machine. The data is shared!
            
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            # zip contains folder `pcqm4m-v2/raw`, so when extracting, the file will go to `self.raw_dir` folder
            os.unlink(path)
            
            path = download_url(self.pos_url, self.raw_dir)
            tar = tarfile.open(path, "r:gz")
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.raw_dir)
            tar.close()
            os.unlink(path)
            
        else:
            from torch_geometric.data.dataset import files_exist

            while not files_exist(self.raw_paths):
                print(f"sleep for RANK {os.environ.get('RANK', 0)} ...")
                time.sleep(3)

    def process(self):
        processes = 4
        data_df = pd.read_csv(os.path.join(self.raw_dir, "data.csv.gz"))
        graph_pos_list = Chem.SDMolSupplier(
            os.path.join(self.raw_dir, "pcqm4m-v2-train.sdf")
        )
        homolumogap_list = data_df["homolumogap"]
        num_3d = len(graph_pos_list)
        num_all = len(homolumogap_list)
        print(f"Totally {num_all} molecules, with {num_3d} having 3D positions!")

        print(
            f"Extracting 3D positions of {num_3d} molecules from SDF files for Training Data..."
        )
        train_data_with_position_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(mol2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
                try:
                    data = Data()
                    homolumogap = homolumogap_list[i]


                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(graph["position"]).to(torch.float32)

                    train_data_with_position_list.append(data)
                except:
                    continue
        print(
            f"Done extracting 3D positions of {len(train_data_with_position_list)}/{num_3d}!"
        )


        data_list = train_data_with_position_list 

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["train"]])
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["valid"]])
        assert all([torch.isnan(data_list[i].y)[0] for i in split_dict["test-dev"]])
        assert all(
            [torch.isnan(data_list[i].y)[0] for i in split_dict["test-challenge"]]
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(os.path.join(self.root, "split_dict.pt"))
        )
        return split_dict


class PyGHamiltonianNablaDFT(Dataset):
    """Pytorch Geometric interface for nablaDFT Hamiltonian datasets.

    .. code-block:: python
        from nablaDFT.dataset import (
            PyGHamiltonianNablaDFT,
        )

        dataset = PyGHamiltonianNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/hamiltonian_databases.json), dataset will be downloaded automatically.

    .. note::
        Hamiltonian matrix for each molecule has different shape. PyTorch Geometric tries to concatenate
        each torch.Tensor in batch, so in order to make batch from data we leave all hamiltonian matrices
        in numpy array form. During train, these matrices will be yield as List[np.array].

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        include_hamiltonian (bool): if True, retrieves full Hamiltonian matrices from database.
        include_overlap (bool): if True, retrieves overlap matrices from database.
        include_core (bool): if True, retrieves core Hamiltonian matrices from database.
        dtype (torch.dtype): defines torch.dtype for energy, positions and forces tensors.
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
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
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
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
        z = torch.tensor(data[0].copy()).long()
        positions = torch.tensor(data[1].copy()).to(self.dtype)
        # see notes
        hamiltonian = data[4].copy()
        if self.include_overlap:
            overlap = data[5].copy()
        else:
            overlap = None
        if self.include_core:
            core = data[6].copy()
        else:
            core = None
        y = torch.from_numpy(data[2].copy()).to(self.dtype)
        forces = torch.from_numpy(data[3].copy()).to(self.dtype)
        data = Data(
            z=z,
            pos=positions,
            y=y,
            forces=forces,
            hamiltonian=hamiltonian,
            overlap=overlap,
            core=core,
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    def download(self) -> None:
        url = dataset_registry.get_dataset_url("hamiltonian", self.dataset_name)
        dataset_etag = dataset_registry.get_dataset_etag("hamiltonian", self.dataset_name)
        download_file(
            url,
            Path(self.raw_paths[0]),
            dataset_etag,
            desc=f"Downloading split: {self.dataset_name}",
        )

    def _get_max_orbitals(self, datapath, dataset_name):
        db_path = os.path.join(datapath, "raw/" + dataset_name + self.db_suffix)
        if not os.path.exists(db_path):
            self.download()
        database = HamiltonianDatabase(db_path)
        max_orbitals = []
        for z in database.Z:
            max_orbitals.append(tuple((int(z), int(orb_num)) for orb_num in database.get_orbitals(z)))
        max_orbitals = tuple(max_orbitals)
        return max_orbitals


class PyGPQCDataset(InMemoryDataset):
    """Pytorch Geometric interface for nablaDFT datasets.

    Based on `MD17 implementation <https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/md17.py>`_.

    .. code-block:: python
        from nablaDFT.dataset import PyGNablaDFT

        dataset = PyGNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/energy_databases.json), dataset will be downloaded automatically.

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    db_suffix = ".lmdb"

    @property
    def raw_file_names(self) -> List[str]:
        return [(f"{self.split}_{self.dataset_name}{self.db_suffix}")]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        print(self.raw_file_names)
        super(PyGPQCDataset, self).__init__(datapath, transform, pre_transform)

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self) -> int:
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGPQCDataset, self).get(idx - self.offsets[data_idx])

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        env_label_3D = lmdb.open(
            self.raw_paths[0],
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )

        with env_label_3D.begin() as txn:
            length = txn.stat()['entries']
            samples = []
            for key, value in tqdm(txn.cursor(), total=length):
                dt = json.loads(value.decode("utf-8"))
                z = np.array(dt['elements'])
                if len(z) == 1:
                    print (key, dt['smiles'])
                    continue
                z = torch.from_numpy(z).long()
                y = torch.from_numpy(np.array([dt['gap']])).float()
                #for positions in dt['input_pos']:
                #    pos = torch.from_numpy(positions).float()
                #    samples.append(Data(z=z, pos=pos, y=y))
                positions = dt['coords_3d']
                pos = torch.from_numpy(np.array(positions).reshape(-1,3)).float()
                samples.append(Data(z=z, pos=pos, y=y))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")
