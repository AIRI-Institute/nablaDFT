import numpy as np
import torch
from torch_geometric.data import Data, Batch


def np_scatter_add(updates, indices, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    np.add.at(target, indices, updates)
    return target


def atoms_list_to_PYG(ase_atoms_list, device):
    """Function to convert ase.Atoms object to PyG data batches.
    Args:
        ase_atoms_list (List[ase.Atoms]): list of ase.Atoms object to convert.
        device (str): task device.
    """
    data = []
    for ase_atoms in ase_atoms_list:
        z = torch.from_numpy(ase_atoms.numbers).long()
        positions = torch.from_numpy(ase_atoms.positions).float()
        data.append(Data(z=z, pos=positions))
    batch = Batch.from_data_list(data).to(device)
    return batch
