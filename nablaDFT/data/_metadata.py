"""Module defines mapping between column names in various datasources."""

import json
from ast import literal_eval
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np


@dataclass
class DatasourceCard:
    """Describes dataset metadata.

    Args:
        desc (Optional[Dict]) - dataset description.
        metadata (Optional[Dict]) - dataset metadata.
        columns (Optional[Dict]) - columns from `data` table.
        _keys_map (Optional[Dict]) - mapping from column names in database to sample's keys.
        _data_dtypes (Optional[Dict]) - mapping from column names in database to data type (e.g. np.float32).
        _data_shapes (Optional[Dict]) - mapping from column names in database to data shape (e.g. (-1, 3)).
    """

    desc: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    _keys_map: Dict[str, str] = field(default_factory=dict)
    _dtypes: Dict[str, np.dtype] = field(default_factory=dict)
    _shapes: Dict[str, Tuple] = field(default_factory=dict)

    def __post_init__(self):
        if self._dtypes is not None:
            self._dtypes = _parse_dtypes(self._dtypes)
        if self._shapes is not None:
            self._shapes = _parse_shapes(self._shapes)

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, "r") as fin:
            return DatasourceCard(**json.loads(fin.read()))


# move to _metadata.py
def _parse_dtypes(dtypes_desc: Dict[str, Union[str, np.dtype]]) -> Dict[str, np.dtype]:
    """Returns dtypes from metadata mapping from sample keys to numpy dtypes."""
    dtypes = {}
    for key, value in dtypes_desc.items():
        if isinstance(value, str):
            if value in ["str", "int", "float"]:
                dtypes[key] = value
            else:
                dtypes[key] = np.dtype(value)
        elif isinstance(value, np.dtype):
            dtypes[key] = value
        elif issubclass(value, np.generic):
            dtypes[key] = value
        else:
            raise ValueError(f"Expected scalar numpy, string, int or float, got {type(value)}")
    return dtypes


# move to _metadata.py
def _parse_shapes(shapes_desc: Dict[str, Union[str, tuple]]) -> Dict[str, tuple]:
    """Returns shapes from metadata mapping from sample keys to numpy dtypes."""
    shapes = {}
    for key, value in shapes_desc.items():
        if isinstance(value, str):
            shapes[key] = literal_eval(value)
        elif isinstance(value, tuple):
            shapes[key] = value
        else:
            raise ValueError(f"Expected tuple or string, got {type(value)}")
    return shapes
