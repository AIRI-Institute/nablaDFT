import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from nablaDFT.dataset import PyGHamiltonianNablaDFT, PyGNablaDFT, HamiltonianDataset


@pytest.mark.dataset
def test_energy_dataset_items(dataset_pyg_params):
    dataset_pyg_params["split"] = "train"
    dataset = PyGNablaDFT(**dataset_pyg_params)
    sample = dataset[0]
    y, pos, z, forces = sample.y, sample.pos, sample.z, sample.forces
    # check types
    assert isinstance(y, torch.Tensor)
    assert isinstance(pos, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    assert isinstance(forces, torch.Tensor)
    # check shapes
    assert y.shape == torch.Size([1])
    assert z.shape == torch.Size([40])
    assert pos.shape == forces.shape == torch.Size([40, 3])
    # check multi-index __getitem__
    sample = dataset[15:30]
    y, pos, z, forces = sample.y, sample.pos, sample.z, sample.forces
    assert y.shape == torch.Size([15])
    assert pos.shape == forces.shape == torch.Size([610, 3])
    assert z.shape == torch.Size([610])


def test_energy_dataset_not_found(dataset_pyg_params):
    dataset_pyg_params["dataset_name"] = "non_existing_dataset"
    with pytest.raises(KeyError) as e_info:
        dataset = PyGNablaDFT(**dataset_pyg_params)


@pytest.mark.dataset
def test_hamiltonian_dataset_items(dataset_hamiltonian_pyg_params):
    dataset_hamiltonian_pyg_params["split"] = "train"
    dataset = PyGHamiltonianNablaDFT(**dataset_hamiltonian_pyg_params)
    sample = dataset[0]
    y, pos, z, forces = sample.y, sample.pos, sample.z, sample.forces
    H, S, C = sample.hamiltonian, sample.overlap, sample.core
    # check types
    assert isinstance(y, torch.Tensor)
    assert isinstance(pos, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    assert isinstance(forces, torch.Tensor)
    assert isinstance(H, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert isinstance(C, np.ndarray)
    # check shapes
    assert y.shape == torch.Size([1])
    assert z.shape == torch.Size([38])
    assert pos.shape == forces.shape == torch.Size([38, 3])
    assert z.shape == torch.Size([38])
    assert H.shape == S.shape == C.shape == (396, 396)
    # check multi-index __getitem__
    subset = dataset[3:15]
    assert isinstance(subset, PyGHamiltonianNablaDFT)
    for sample in subset:
        y, pos, z, forces = sample.y, sample.pos, sample.z, sample.forces
        H, S, C = sample.hamiltonian, sample.overlap, sample.core
        assert y.shape == torch.Size([1])
        assert pos.shape == forces.shape
        assert H.shape == S.shape == C.shape


def test_hamiltonian_torch_dataset(dataset_hamiltonian_torch_params):
    dataset = HamiltonianDataset(**dataset_hamiltonian_torch_params)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn, shuffle=False)
    database = dataset.database
    shape_gt = sum([database[idx][4].shape[0] for idx in [0, 1]])
    batch = next(iter(dataloader))
    assert batch['full_hamiltonian'].shape == torch.Size([shape_gt, shape_gt])
    assert batch['full_hamiltonian'].shape == batch['overlap_matrix'].shape == batch['core_hamiltonian'].shape
