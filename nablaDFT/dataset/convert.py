from functools import singledispatch
from typing import Dict, Tuple

import numpy as np
import torch
import torch_geometric as pyg

NUMPY_TO_TORCH = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


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
