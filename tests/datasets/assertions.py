import numpy as np
import torch


def assert_shapes_hamiltonian(batch):
    assert batch.forces.shape == batch.pos.shape
    assert isinstance(batch.hamiltonian, list)
    assert isinstance(batch.overlap, list)
    assert isinstance(batch.core, list)
    for H, S, C in zip(batch.hamiltonian, batch.overlap, batch.core):
        assert isinstance(H, np.ndarray) and isinstance(S, np.ndarray) and isinstance(C, np.ndarray)
        assert H.shape == S.shape == C.shape


def assert_shapes_spk(batch):
    assert batch['energy'].shape == batch['_idx'].shape
    assert batch['forces'].shape == batch['_positions'].shape == torch.Size([batch['_atomic_numbers'].shape[0], 3])
