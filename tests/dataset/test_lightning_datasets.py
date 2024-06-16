from math import ceil

import pytest
import torch

from nablaDFT.dataset import PyGHamiltonianDataModule, PyGNablaDFTDataModule
from .assertions import assert_shapes_hamiltonian


@pytest.mark.dataset
def test_lightning_energy_datamodule_fit_dataloader(dataset_lightning_params):
    dataset = PyGNablaDFTDataModule(**dataset_lightning_params)
    dataset.setup("fit")
    dataloader_train, dataloader_val = (
        dataset.train_dataloader(),
        dataset.val_dataloader(),
    )
    assert len(dataloader_train) == ceil(
        len(dataset.dataset_train) / dataset.dataloader_kwargs["batch_size"]
    )
    assert len(dataloader_val) == ceil(
        len(dataset.dataset_val) / dataset.dataloader_kwargs["batch_size"]
    )
    train_batch = next(iter(dataloader_train))
    val_batch = next(iter(dataloader_train))
    for batch in [train_batch, val_batch]:
        assert batch.y.shape == torch.Size([dataset.dataloader_kwargs["batch_size"]])
        assert batch.forces.shape == batch.pos.shape


@pytest.mark.dataset
def test_lightning_energy_datamodule_test_dataloader(dataset_lightning_params):
    dataset_lightning_params["batch_size"] = 100
    dataset = PyGNablaDFTDataModule(**dataset_lightning_params)
    dataset.setup("test")
    dataloader_test = dataset.test_dataloader()
    batch = next(iter(dataloader_test))
    assert batch.y.shape == torch.Size([100])
    assert batch.pos.shape == batch.forces.shape == torch.Size([4198, 3])
    assert batch.z.shape == torch.Size([4198])


@pytest.mark.dataset
def test_lightning_energy_datamodule_predict_dataloader(dataset_lightning_params):
    dataset_lightning_params["batch_size"] = 100
    dataset = PyGNablaDFTDataModule(**dataset_lightning_params)
    dataset.setup("predict")
    dataloader_predict = dataset.predict_dataloader()
    batch = next(iter(dataloader_predict))
    assert batch.y.shape == torch.Size([100])
    assert batch.pos.shape == batch.forces.shape == torch.Size([4198, 3])
    assert batch.z.shape == torch.Size([4198])


@pytest.mark.dataset
def test_lightning_hamiltonian_datamodule_fit_dataloader(
    dataset_hamiltonian_lightning_params,
):
    dataset = PyGHamiltonianDataModule(**dataset_hamiltonian_lightning_params)
    dataset.setup("fit")
    dataloader_train, dataloader_val = (
        dataset.train_dataloader(),
        dataset.val_dataloader(),
    )
    train_batch = next(iter(dataloader_train))
    val_batch = next(iter(dataloader_train))
    for batch in [train_batch, val_batch]:
        assert_shapes_hamiltonian(batch)


@pytest.mark.dataset
def test_lightning_hamiltonian_datamodule_test_dataloader(
    dataset_hamiltonian_lightning_params,
):
    dataset = PyGHamiltonianDataModule(**dataset_hamiltonian_lightning_params)
    dataset.setup("test")
    dataloader = dataset.test_dataloader()
    batch = next(iter(dataloader))
    assert_shapes_hamiltonian(batch)


@pytest.mark.dataset
def test_lightning_hamiltonian_datamodule_predict_dataloader(
    dataset_hamiltonian_lightning_params,
):
    dataset = PyGHamiltonianDataModule(**dataset_hamiltonian_lightning_params)
    dataset.setup("predict")
    dataloader = dataset.predict_dataloader()
    batch = next(iter(dataloader))
    assert_shapes_hamiltonian(batch)
