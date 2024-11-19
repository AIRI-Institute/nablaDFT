from pathlib import Path

import nablaDFT
import pytest
from nablaDFT.data import EnergyDatabase, SQLite3Database
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
def overlap_metadata():
    json_path = Path(nablaDFT.__path__[0]) / "data/_metadata/nabla_overlap.json"
    return DatasourceCard.from_json(json_path)


@pytest.fixture()
def datapath_energy():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/test_data/energy.db"
    return datapath


@pytest.fixture()
def datapath_hamiltonian():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/test_data/hamiltonian.db"
    return datapath


@pytest.fixture()
def datapath_overlap():
    datapath = Path(nablaDFT.__path__[0]) / "../tests/test_data/overlap.db"
    return datapath


@pytest.fixture()
def test_energy_db(datapath_energy, energy_metadata):
    return EnergyDatabase(datapath_energy, energy_metadata)


@pytest.fixture()
def test_hamiltonian_db(datapath_hamiltonian, hamiltonian_metadata):
    return SQLite3Database(datapath_hamiltonian, hamiltonian_metadata)


@pytest.fixture()
def test_datasources_list(datapath_hamiltonian, datapath_overlap, hamiltonian_metadata, overlap_metadata):
    return [
        SQLite3Database(datapath_hamiltonian, hamiltonian_metadata),
        SQLite3Database(datapath_overlap, overlap_metadata),
    ]
