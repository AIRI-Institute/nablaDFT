from pathlib import Path

import nablaDFT
import pytest
from nablaDFT.data._metadata import DatasourceCard


@pytest.fixture()
def energy_metadata():
    json_path = Path(nablaDFT.__path__[0]) / "data/_metadata/nabla_energy.json"
    return DatasourceCard.from_json(json_path)


@pytest.fixture()
def hamiltonian_metadata():
    json_path = Path(nablaDFT.__path__[0]) / "data/_metadata/nabla_hamiltonian.json"
    return DatasourceCard.from_json(json_path)


@pytest.fixture()
def datapath_energy():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/test_data/energy.db"
    return datapath


@pytest.fixture()
def datapath_hamiltonian():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/test_data/hamiltonian.db"
    return datapath
