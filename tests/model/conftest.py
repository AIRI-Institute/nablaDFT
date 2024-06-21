from pathlib import Path

import nablaDFT
import pytest
from nablaDFT.dataset import PyGHamiltonianNablaDFT, PyGNablaDFT
from schnetpack.data import ASEAtomsData
from schnetpack.transform import ASENeighborList, CastTo32


@pytest.fixture()
def dataset_pyg():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/data"
    dataset_name = "test_database"
    params = {
        "datapath": str(datapath),
        "dataset_name": dataset_name,
        "transform": None,
        "pre_transform": None,
    }
    dataset = PyGNablaDFT(**params)
    return dataset


@pytest.fixture()
def dataset_spk():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/data/raw/test_database.db"
    transforms = [
        ASENeighborList(cutoff=5.0),
        CastTo32(),
    ]
    dataset = ASEAtomsData(datapath, transforms=transforms)
    return dataset


@pytest.fixture()
def dataset_hamiltonian():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/data"
    dataset_name = "test_hamiltonian_database"
    params = {
        "datapath": str(datapath),
        "dataset_name": dataset_name,
        "include_hamiltonian": True,
        "include_overlap": False,
        "include_core": False,
        "transform": None,
        "pre_transform": None,
    }
    dataset = PyGHamiltonianNablaDFT(**params)
    return dataset
