import pickle
from copy import deepcopy
from pathlib import Path

import nablaDFT
import pytest

BATCH_SIZE = 8
NUM_WORKERS = 4


@pytest.fixture()
def hamiltonian_orbitals():
    path = Path(nablaDFT.__path__[0]) / "../tests/data/orbitals.pkl"
    with open(path, "rb") as fin:
        orbitals = pickle.loads(fin.read())
    return orbitals


@pytest.fixture()
def dataset_pyg_params():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/data"
    dataset_name = "test_database"
    params = {
        "datapath": str(datapath),
        "dataset_name": dataset_name,
        "transform": None,
        "pre_transform": None,
    }
    return params


@pytest.fixture()
def dataset_hamiltonian_pyg_params():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/data"
    dataset_name = "test_hamiltonian_database"
    params = {
        "datapath": str(datapath),
        "dataset_name": dataset_name,
        "include_hamiltonian": True,
        "include_overlap": True,
        "include_core": True,
        "transform": None,
        "pre_transform": None,
    }
    return params


@pytest.fixture()
def dataset_hamiltonian_db_params():
    db_path = Path(nablaDFT.__path__[0]) / ".." / "tests/data/raw/test_hamiltonian_database.db"
    return str(db_path)


@pytest.fixture()
def dataset_hamiltonian_torch_params():
    db_path = Path(nablaDFT.__path__[0]) / ".." / "tests/data/raw/test_hamiltonian_database.db"
    params = {
        "filepath": str(db_path),
        "max_batch_orbitals": 1200,
        "max_batch_atoms": 150,
        "max_squares": 4802,
    }
    return params


@pytest.fixture()
def dataset_lightning_params():
    train_size = 0.9
    val_size = 0.1
    datapath = Path(nablaDFT.__path__[0]) / "../tests/data"
    dataset_name = "test_database"
    params = {
        "root": str(datapath),
        "dataset_name": dataset_name,
        "train_size": train_size,
        "val_size": val_size,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": True,
    }
    return params


@pytest.fixture()
def dataset_hamiltonian_lightning_params(dataset_lightning_params):
    params = deepcopy(dataset_lightning_params)
    params["dataset_name"] = "test_hamiltonian_database"
    params["include_hamiltonian"] = True
    params["include_core"] = True
    params["include_overlap"] = True
    return params
