from functools import singledispatch
from typing import Dict, Tuple

import numpy as np
import torch
import torch_geometric as pyg


@singledispatch
def convert(type_cls, data):
    raise TypeError(f"Cannot convert {type(data)} to {type_cls}")


@convert.register
def _(type_cls: torch.Tensor, data: Dict[str, np.array]) -> Dict[torch.Tensor]:
    for key, value in data.items():
        data[key] = torch.from_numpy(value.copy())
    return data


@convert.register
def _(type_cls: pyg.data.Data, data: Dict[np.array]) -> pyg.data.Data:
    data = convert(torch.Tensor, data)
    return type_cls(**data)


@singledispatch
def convert_nabla_energy(type_cls: torch.Tensor, data: Dict[str, np.array]) -> Dict[torch.Tensor]:
    pass


@singledispatch
def convert_nabla_energy(type_cls: pyg.data.Data, data: Dict[np.array]) -> pyg.data.Data:
    data = convert_nabla_energy(torch.Tensor, data)
    return type_cls(**data)
