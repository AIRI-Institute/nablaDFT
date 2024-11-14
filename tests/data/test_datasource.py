from pathlib import Path

import numpy as np
import pytest
from nablaDFT.data import EnergyDatabase, SQLite3Database
from nablaDFT.data._metadata import DatasourceCard


@pytest.mark.data
def test_energy_db_get(energy_metadata: DatasourceCard, datapath_energy: Path):
    """Base elements retireval test for energy databases."""
    db = EnergyDatabase(datapath_energy, energy_metadata)
    assert len(db) == 100
    # single element
    sample = db[0]
    assert isinstance(sample, dict)
    assert list(sample.keys()) == ["z", "y", "pos", "forces"]
    for key in sample.keys():
        assert isinstance(sample[key], np.ndarray)
    # slicing
    samples = db[:10]
    assert isinstance(samples, list)
    assert len(samples) == 10
    for sample in samples:
        assert list(sample.keys()) == ["z", "y", "pos", "forces"]
        for key in sample.keys():
            assert isinstance(sample[key], np.ndarray)
    # indexing with list
    samples = db[[2, 15, 78]]
    assert isinstance(samples, list)
    assert len(samples) == 3
    for sample in samples:
        assert list(sample.keys()) == ["z", "y", "pos", "forces"]
        for key in sample.keys():
            assert isinstance(sample[key], np.ndarray)


@pytest.mark.data
def test_sqlite_db_get(hamiltonian_metadata: DatasourceCard, datapath_hamiltonian: Path):
    """Base elements retireval test for SQLite3-based datasources."""
    db = SQLite3Database(datapath_hamiltonian, hamiltonian_metadata)
    # single element
    assert len(db) == 25
    sample = db[0]
    assert isinstance(sample, dict)
    assert sample.keys() == hamiltonian_metadata._keys_map.keys()
    for key in sample.keys():
        dtype = np.dtype(db._dtypes[db._keys_map[key]])
        assert sample[key].dtype == dtype
    # slicing
    samples = db[:5]
    assert isinstance(samples, list)
    assert len(samples) == 5
    for sample in samples:
        assert sample.keys() == hamiltonian_metadata._keys_map.keys()
        for key in sample.keys():
            dtype = np.dtype(db._dtypes[db._keys_map[key]])
            assert sample[key].dtype == dtype
    # indexing with list
    samples = db[[2, 15, 8]]
    assert isinstance(samples, list)
    assert len(samples) == 3
    for sample in samples:
        assert sample.keys() == hamiltonian_metadata._keys_map.keys()
        for key in sample.keys():
            dtype = np.dtype(db._dtypes[db._keys_map[key]])
            assert sample[key].dtype == dtype


@pytest.mark.data
def test_sqlite_db_get_no_meta(datapath_hamiltonian):
    """Base elements retireval test for SQLite3-based datasources without metadata."""
    db = SQLite3Database(datapath_hamiltonian)
    # check that columns is all cols from data table
    assert db.columns == ["Z", "R", "E", "F", "H", "S"]
