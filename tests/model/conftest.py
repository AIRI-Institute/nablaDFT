from pathlib import Path

import pytest

import nablaDFT
from nablaDFT.dataset import PyGNablaDFT


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
