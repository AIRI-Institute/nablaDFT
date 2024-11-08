# TODO: add _parse_* to DatasetCard.__post_init__()
"""Module defines mapping between column names in various datasources."""

import json
import pprint
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DatasetCard:
    """Describes dataset metadata.

    Args:
        desc (Optional[Dict]) - dataset description. Could be empty.
        metadata (Optional[Dict]) - dataset metadata. Could be empty. Must contain calculation methods.
        _keys_map (Optional[Dict]) - mapping from column names in database to sample's keys.
        _data_dtypes (Optional[Dict]) - mapping from column names in database to data type (e.g. np.float32).
        _data_shapes (Optional[Dict]) - mapping from column names in database to data shape (e.g. (-1, 3)).
    """

    desc: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    _keys_map: List = field(default_factory=list)
    _data_dtypes: Dict[str, np.dtype] = field(default_factory=dict)
    _data_shapes: Dict[str, Tuple] = field(default_factory=dict)

    def __post_init__(self):
        if self._keys_map is not None:
            self._keys_map = _parse_key_map(self._keys_map)
        if self._data_dtypes is not None:
            self._data_dtypes = _parse_dtypes(self._data_dtypes)
        if self._data_shapes is not None:
            self._data_shapes = _parse_shapes(self._data_shapes)

    def __str__(self) -> str:
        desc = pprint.pformat(self.desc)
        metadata = pprint.pformat(self.metadata)
        return f"{desc}\n{metadata}"

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, "r") as fin:
            return DatasetCard(**json.loads(fin.read()))


def _parse_key_map(mapping: Dict[str, str]) -> Tuple[List]:
    """Returns parsed from metadata mapping from datasource keys to sample keys."""
    sample_keys, tables, db_keys = [], [], []
    for key, value in mapping.items():
        sample_keys.append(key)
        if "." in value:
            table, db_key = value.split(".")
            db_keys.append(db_key)
            tables.append(table)
        else:
            db_keys.append(value)
    if tables:
        return sample_keys, tables, db_keys
    return sample_keys, db_keys


# move to _metadata.py
def _parse_dtypes(dtypes_desc: Dict[str, str]) -> Dict[str, np.dtype]:
    """Returns dtypes from metadata mapping from sample keys to numpy dtypes."""
    dtypes = {}
    for key, value in dtypes_desc.items():
        dtypes[key] = np.dtype(value)
    return dtypes


# move to _metadata.py
def _parse_shapes(shapes_desc: Dict[str, str]) -> Dict[str, tuple]:
    """Returns shapes from metadata mapping from sample keys to numpy dtypes."""
    shapes = {}
    for key, value in shapes_desc.items():
        shapes[key] = literal_eval(value)
    return shapes
