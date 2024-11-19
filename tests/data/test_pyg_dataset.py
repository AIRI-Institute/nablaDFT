import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from nablaDFT.data import Datasource, PyGDataset, SQLite3Database
from nablaDFT.data._collate import SQUARE_SHAPED_KEYS
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData


@pytest.mark.data
@pytest.mark.parametrize("example_datasource", ["test_energy_db", "test_hamiltonian_db"])
def test_pyg_dataset_get(request, example_datasource):
    datasource: Datasource = request.getfixturevalue(example_datasource)
    dataset = PyGDataset(datasource, in_memory=False)
    # single element
    datasource_sample = datasource[3]
    sample: BaseData = dataset[3]
    assert isinstance(sample, BaseData)
    for key in sample.keys():
        assert isinstance(getattr(sample, key), torch.Tensor)
    for key in datasource._keys_map.keys():
        assert np.allclose(sample[key].numpy(), datasource_sample[key])
    # slice
    datasource_samples = datasource[0:3]
    samples: BaseData = dataset[0:3]
    assert isinstance(samples, list)
    for sample, datasource_sample in zip(samples, datasource_samples):
        assert isinstance(sample, BaseData)
        for key in sample.keys():
            assert isinstance(getattr(sample, key), torch.Tensor)
        for key in datasource._keys_map.keys():
            assert np.allclose(sample[key].numpy(), datasource_sample[key])
    # list index
    datasource_samples = datasource[[3, 5, 15]]
    samples: BaseData = dataset[[3, 5, 15]]
    assert isinstance(samples, list)
    for sample, datasource_sample in zip(samples, datasource_samples):
        assert isinstance(sample, BaseData)
        for key in sample.keys():
            assert isinstance(getattr(sample, key), torch.Tensor)
        for key in datasource._keys_map.keys():
            assert np.allclose(sample[key].numpy(), datasource_sample[key])


@pytest.mark.data
@pytest.mark.parametrize("in_memory", [True, False])
def test_pyg_dataset_get_multiple_datasources(
    in_memory, monkeypatch, tmp_path, datapath_overlap, datapath_hamiltonian, hamiltonian_metadata, overlap_metadata
):
    monkeypatch.chdir(tmp_path)
    shutil.copy2(datapath_overlap, Path(tmp_path) / datapath_overlap.name)
    shutil.copy2(datapath_hamiltonian, Path(tmp_path) / datapath_hamiltonian.name)
    db_H = SQLite3Database(Path(tmp_path) / datapath_hamiltonian.name, hamiltonian_metadata)
    db_S = SQLite3Database(Path(tmp_path) / datapath_overlap.name, overlap_metadata)
    dataset = PyGDataset([db_H, db_S], in_memory=in_memory)
    keys = list(set.union(set(db_H._keys_map.keys()) & set(db_S._keys_map.keys())))
    # single sample
    sample = dataset[0]
    assert isinstance(sample, BaseData)
    for key in keys:
        assert isinstance(getattr(sample, key), torch.Tensor)
    # slice
    samples = dataset[3:5]
    for sample in samples:
        assert isinstance(sample, BaseData)
        for key in keys:
            assert isinstance(getattr(sample, key), torch.Tensor)
    # list index
    samples = dataset[[2, 5, 16]]
    for sample in samples:
        assert isinstance(sample, BaseData)
        for key in keys:
            assert isinstance(getattr(sample, key), torch.Tensor)
    monkeypatch.undo()


@pytest.mark.data
def test_in_memory_pyg_dataset_get(datapath_hamiltonian, hamiltonian_metadata, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy2(datapath_hamiltonian, Path(tmp_path) / datapath_hamiltonian.name)
    datasource = SQLite3Database(Path(tmp_path) / datapath_hamiltonian.name, hamiltonian_metadata)
    dataset = PyGDataset(datasource, in_memory=True)
    assert (Path(tmp_path) / "processed").exists()
    assert (Path(tmp_path) / "processed").glob("*")
    assert dataset.data
    assert dataset.slices
    # single element
    datasource_sample = datasource[3]
    sample = dataset[3]
    assert isinstance(sample, BaseData)
    for key in sample.keys():
        assert isinstance(getattr(sample, key), torch.Tensor)
    for key in datasource._keys_map.keys():
        assert np.allclose(sample[key].numpy(), datasource_sample[key])
    # slice
    datasource_samples = datasource[0:3]
    samples: BaseData = dataset[0:3]
    assert isinstance(samples, list)
    for sample, datasource_sample in zip(samples, datasource_samples):
        assert isinstance(sample, BaseData)
        for key in sample.keys():
            assert isinstance(getattr(sample, key), torch.Tensor)
        for key in datasource._keys_map.keys():
            assert np.allclose(sample[key].numpy(), datasource_sample[key])
    # list index
    datasource_samples = datasource[[3, 5, 15]]
    samples: BaseData = dataset[[3, 5, 15]]
    assert isinstance(samples, list)
    for sample, datasource_sample in zip(samples, datasource_samples):
        assert isinstance(sample, BaseData)
        for key in sample.keys():
            assert isinstance(getattr(sample, key), torch.Tensor)
        for key in datasource._keys_map.keys():
            assert np.allclose(sample[key].numpy(), datasource_sample[key])
    monkeypatch.undo()


@pytest.mark.data
@pytest.mark.parametrize("example_datasource", ["test_energy_db", "test_hamiltonian_db"])
def test_pyg_collation(request, example_datasource):
    datasource = request.getfixturevalue(example_datasource)
    dataset = PyGDataset(datasource, in_memory=False)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        assert isinstance(batch, Batch)
        assert len(getattr(batch, "ptr")) is not None
        assert getattr(batch, "batch") is not None
        for key in datasource._keys_map.keys():
            field = getattr(batch, key)
            if key in SQUARE_SHAPED_KEYS:
                assert len(getattr(batch, key)) == (len(getattr(batch, "ptr")) - 1)
                assert isinstance(field, list)
                for element in field:
                    assert element.shape[0] == element.shape[1]
