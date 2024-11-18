from pathlib import Path

import numpy as np
import pytest
import torch
from nablaDFT.data import Datasource, PyGDataset
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
@pytest.mark.parametrize("example_datasource", ["test_energy_db", "test_hamiltonian_db"])
def test_pyg_dataset_get_multiple_datasources(request, example_datasource):
    datasource = request.getfixturevalue(example_datasource)
    dataset = PyGDataset(datasource, in_memory=False)


@pytest.mark.parametrize("example_datasource", ["test_energy_db", "test_hamiltonian_db"])
@pytest.mark.data
def test_in_memory_pyg_dataset_get(request, example_datasource, monkeypatch, tmp_path):
    datasource = request.getfixturevalue(example_datasource)
    monkeypatch.chdir(tmp_path)
    dataset = PyGDataset(datasource, in_memory=True)
    assert (Path(tmp_path) / "processed").exists()
    assert (Path(tmp_path) / "processed").glob("*")
    # single element
    # slice
    # list index
    monkeypatch.undo()


@pytest.mark.data
@pytest.mark.parametrize("example_datasource", ["test_energy_db", "test_hamiltonian_db"])
def test_in_memory_pyg_dataset_get_multiple_datasources(request, example_datasource):
    datasource = request.getfixturevalue(example_datasource)
    dataset = PyGDataset(datasource, in_memory=False)


@pytest.mark.data
@pytest.mark.parametrize("example_datasource", ["test_energy_db", "test_hamiltonian_db"])
def test_pyg_collation(request, example_datasource):
    datasource = request.getfixturevalue(example_datasource)
    dataset = PyGDataset(datasource, in_memory=False)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        assert isinstance(batch, Batch)
        assert len(getattr(batch, "ptr")) == 3
        assert len(getattr(batch, "ptr")) == 3
