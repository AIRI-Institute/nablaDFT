from functools import singledispatch
from typing import Dict, Tuple

import numpy as np
import torch
import torch_geometric as pyg


@singledispatch
def convert_nabla_hamiltonian(type_cls, data):
    raise TypeError(f"Cannot convert {type(data)} to {type_cls}")


@convert.register
def convert_nabla_hamiltonian(type_cls: torch.Tensor, data: Dict[str, np.array]) -> Dict[torch.Tensor]:
    for key, value in data.items():
        np_dtype = value.dtype
        data[key] = torch.from_numpy(value.copy()).to(NUMPY_TO_TORCH[np_dtype])
    return data


@convert.register
def convert_nabla_hamiltonian(type_cls: pyg.data.Data, data: Dict[np.array]) -> pyg.data.Data:
    data = convert_nabla_hamiltonian(torch.Tensor, data)
    return type_cls(**data)


@singledispatch
def convert_nabla_energy(type_cls: torch.Tensor, data: Dict[str, np.array]) -> Dict[torch.Tensor]:
    pass


@singledispatch
def convert_nabla_energy(type_cls: pyg.data.Data, data: Dict[np.array]) -> pyg.data.Data:
    data = convert_nabla_energy(torch.Tensor, data)
    return type_cls(**data)
