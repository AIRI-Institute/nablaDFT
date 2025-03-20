import shutil
from pathlib import Path

import numpy as np
import pytest
from nablaDFT.data import EnergyDatabase, SQLite3Database
from nablaDFT.data._convert import _default_dtypes, _default_shapes
from nablaDFT.data.metadata import DatasourceCard


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
        if db._shapes.get(key, None):
            assert sample[key].shape == _default_shapes[key]
    # slicing
    samples = db[:5]
    assert isinstance(samples, list)
    assert len(samples) == 5
    for sample in samples:
        assert sample.keys() == hamiltonian_metadata._keys_map.keys()
        for key in sample.keys():
            dtype = np.dtype(db._dtypes[db._keys_map[key]])
            assert sample[key].dtype == dtype
            if db._shapes.get(key, None):
                assert sample[key].shape == _default_shapes[key]
    # indexing with list
    samples = db[[2, 15, 8]]
    assert isinstance(samples, list)
    assert len(samples) == 3
    for sample in samples:
        assert sample.keys() == hamiltonian_metadata._keys_map.keys()
        for key in sample.keys():
            dtype = np.dtype(db._dtypes[db._keys_map[key]])
            assert sample[key].dtype == dtype
            if db._shapes.get(key, None):
                assert sample[key].shape == _default_shapes[key]


@pytest.mark.data
def test_sqlite_db_get_no_meta(datapath_hamiltonian):
    """Base elements retireval test for SQLite3-based datasources without metadata."""
    db = SQLite3Database(datapath_hamiltonian)
    # check that columns is all cols from data table
    assert db.columns == ["Z", "R", "E", "F", "H"]
    assert [*db._keys_map.keys()] == [*db._keys_map.values()]
    sample = db[0]
    assert isinstance(sample, dict)
    assert sample.keys() == db._keys_map.keys()
    for key in sample.keys():
        dtype = np.dtype(_default_dtypes.get(key, None))
        assert sample[key].dtype == dtype
        if _default_shapes.get(key, None):
            # check axis number
            assert len(sample[key].shape) == len(_default_shapes[key])
    # slicing
    samples = db[:5]
    assert isinstance(samples, list)
    assert len(samples) == 5
    for sample in samples:
        assert sample.keys() == db._keys_map.keys()
        for key in sample.keys():
            dtype = np.dtype(_default_dtypes.get(key, None))
            assert sample[key].dtype == dtype
            if _default_shapes.get(key, None):
                assert len(sample[key].shape) == len(_default_shapes[key])
    # indexing with list
    samples = db[[2, 15, 8]]
    assert isinstance(samples, list)
    assert len(samples) == 3
    for sample in samples:
        assert sample.keys() == db._keys_map.keys()
        for key in sample.keys():
            dtype = np.dtype(_default_dtypes.get(key, None))
            assert sample[key].dtype == dtype
            if _default_shapes.get(key, None):
                assert len(sample[key].shape) == len(_default_shapes[key])


@pytest.mark.data
def test_sqlite_db_create(monkeypatch, tmp_path):
    """Test database creation from minimal metadata."""
    monkeypatch.chdir(tmp_path)
    # case 1
    metadata = {
        "_dtypes": _default_dtypes,
        "columns": ["E", "R", "F", "Z"],
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    new_db = SQLite3Database("test.db", DatasourceCard(**metadata))
    assert "data" in new_db._get_tables_list()
    assert len(new_db) == 0
    assert new_db.columns == metadata["columns"]
    assert [*new_db._keys_map.keys()] == [*new_db._keys_map.values()]
    assert new_db._keys_map == {key: key for key in metadata["columns"]}
    # case 2
    metadata = {
        "_dtypes": _default_dtypes,
        "columns": ["E", "R", "F", "Z"],
        "_keys_map": {"y": "E", "pos": "R", "forces": "F", "numbers": "Z"},
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    new_db = SQLite3Database("test.db", DatasourceCard(**metadata))
    assert new_db._keys_map == metadata["_keys_map"]
    monkeypatch.undo()


@pytest.mark.data
def test_sqlite_db_create_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # no metadata
    with pytest.raises(ValueError):
        SQLite3Database("test.db")
    # dtypes not specified
    metadata = {
        "columns": ["E", "R", "F", "Z"],
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    with pytest.raises(ValueError):
        SQLite3Database("test.db", DatasourceCard(**metadata))
    # shapes not specified
    metadata = {
        "columns": ["E", "R", "F", "Z"],
        "_dtypes": {"E": np.float32, "R": np.float32, "F": np.float32, "Z": np.int32},
    }
    with pytest.raises(ValueError):
        SQLite3Database("test.db", DatasourceCard(**metadata))
    # columns not specified
    metadata = {
        "_dtypes": {"E": np.float32, "R": np.float32, "F": np.float32, "Z": np.int32},
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    with pytest.raises(ValueError):
        SQLite3Database("test.db", DatasourceCard(**metadata))
    monkeypatch.undo()


@pytest.mark.data
def test_sqlite_db_open_fail(monkeypatch, tmp_path, datapath_hamiltonian):
    monkeypatch.chdir(tmp_path)
    shutil.copy(datapath_hamiltonian, Path(tmp_path) / "test.db")
    temp_db = SQLite3Database(Path(tmp_path) / "test.db")
    with temp_db._get_connection() as conn:
        conn.execute("DROP TABLE data")
    # reopen db, check exception
    with pytest.raises(ValueError):
        SQLite3Database(Path(tmp_path) / "test.db")
    monkeypatch.undo()


@pytest.mark.data
def test_sqlite_db_insert(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    metadata = {
        "_dtypes": {"E": np.float32, "R": np.float32, "F": np.float32, "Z": np.int32},
        "columns": ["E", "R", "F", "Z"],
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    new_db = SQLite3Database("test.db", DatasourceCard(**metadata))
    new_data = [
        {
            "E": np.random.rand(1),
            "R": np.random.rand(25, 3),
            "F": np.random.rand(25, 3),
            "Z": np.random.randint(1, size=25),
        }
        for _ in range(10)
    ]
    # write single dict
    new_db.write(new_data[0])
    assert len(new_db) == 1
    sample = new_db[0]
    for key in new_data[0].keys():
        assert np.allclose(new_data[0][key], sample[key])
    # write list of dicts
    new_db.write(new_data[1:])
    assert len(new_db) == 10
    for i in range(10):
        sample = new_db[i]
        for key in new_data[0].keys():
            assert np.allclose(new_data[i][key], sample[key])
    monkeypatch.undo()


@pytest.mark.data
def test_sqlite_db_insert_update_fail(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    metadata = {
        "_dtypes": {"E": np.float32, "R": np.float32, "F": np.float32, "Z": np.int32},
        "columns": ["E", "R", "F", "Z"],
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    new_db = SQLite3Database("test.db", DatasourceCard(**metadata))
    new_data = {
        "E": np.random.rand(1),
        "R": np.random.rand(25, 3),
        "Z": np.random.randint(1, size=25),
    }
    with pytest.raises(ValueError):
        new_db.write(new_data)
    with pytest.raises(ValueError):
        new_db.update(new_data, 0)
    # insufficient keys
    monkeypatch.undo()


@pytest.mark.data
def test_sqlite_db_update(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    metadata = {
        "_dtypes": {"E": np.float32, "R": np.float32, "F": np.float32, "Z": np.int32},
        "columns": ["E", "R", "F", "Z"],
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    new_db = SQLite3Database("test.db", DatasourceCard(**metadata))
    new_data = [
        {
            "E": np.random.rand(1),
            "R": np.random.rand(25, 3),
            "F": np.random.rand(25, 3),
            "Z": np.random.randint(1, size=25),
        }
        for _ in range(10)
    ]
    new_db.write(new_data)
    update_data = {
        "E": np.random.rand(1),
        "R": np.random.rand(25, 3),
        "F": np.random.rand(25, 3),
        "Z": np.random.randint(1, size=25),
    }
    # update single row
    new_db.update(update_data, 4)
    sample = new_db[4]
    for key in sample.keys():
        assert np.allclose(sample[key], update_data[key])
    # update with slice
    update_data = [
        {
            "E": np.random.rand(1),
            "R": np.random.rand(25, 3),
            "F": np.random.rand(25, 3),
            "Z": np.random.randint(1, size=25),
        }
        for _ in range(3)
    ]
    new_db.update(update_data, slice(0, 3, 1))
    for i in [0, 1, 2]:
        sample = new_db[i]
        for key in sample.keys():
            assert np.allclose(sample[key], update_data[i][key])
    # update with list idx
    update_data = [
        {
            "E": np.random.rand(1),
            "R": np.random.rand(25, 3),
            "F": np.random.rand(25, 3),
            "Z": np.random.randint(1, size=25),
        }
        for _ in range(3)
    ]
    new_db.update(update_data, [3, 4, 5])
    for i, idx in enumerate([3, 4, 5]):
        sample = new_db[idx]
        for key in sample.keys():
            assert np.allclose(sample[key], update_data[i][key])
    monkeypatch.undo()


@pytest.mark.data
def test_sqlite_db_delete(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    metadata = {
        "_dtypes": {"E": np.float32, "R": np.float32, "F": np.float32, "Z": np.int32},
        "columns": ["E", "R", "F", "Z"],
        "_shapes": {"R": (-1, 3), "F": (-1, 3)},
    }
    new_db = SQLite3Database("test.db", DatasourceCard(**metadata))
    new_data = [
        {
            "E": np.random.rand(1),
            "R": np.random.rand(25, 3),
            "F": np.random.rand(25, 3),
            "Z": np.random.randint(1, size=25),
        }
        for _ in range(10)
    ]
    new_db.write(new_data)
    # delete one row
    new_db.delete(9)
    assert len(new_db) == 9
    # delete slice
    new_db.delete(slice(5, 9, 1))
    assert len(new_db) == 5
    # delete with index list
    new_db.delete([0, 1, 2, 3, 4])
    assert len(new_db) == 0
    monkeypatch.undo()
