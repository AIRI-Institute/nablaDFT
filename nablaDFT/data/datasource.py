"""Module describes interfaces for datasources, used for model training.

Examples:
--------
.. code-block:: python
    from nablaDFT.dataset import (
        DataSource,
    )

    >>> datasource = EnergyDatabase("path/to/database")
    >>> datasource[0]
    >>> {
    'z': array([...], dtype=int32),
    'y': array(...),
    'pos': array([...]),
    'forces': array([...]),
    }
"""

import math
import pathlib
from ast import literal_eval
from typing import Dict, List, Optional, Tuple, Union

import apsw  # way faster than sqlite3
import ase
import numpy as np
from ase import Atoms

from ._metadata import DatasetCard


class EnergyDatabase:
    """Database interface for energy data.

    Wraps the ASE sqlite3 database for training NNPs.
    For historical reasons we use ASE databases for energy datasets.

    Args:
        filepath (pathlib.Path): path to existing database or path for new database.
        metadata (Optional[DatasetCard]): optional, dataset metadata.
    """

    type = "db"  # only sqlite3 databases

    def __init__(self, filepath: Union[pathlib.Path, str], metadata: Optional[DatasetCard] = None) -> None:
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        self.filepath = filepath.absolute()

        self._db: ase.db.sqlite.SQLite3Database = ase.db.connect(self.filepath, type=self.type)

        # parse sample schema and metadata
        if metadata is not None:
            self.name = metadata.name
            self.metadata = metadata.metadata

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Returns unpacked element from ASE database.

        Args:
            idx (int): index of row to return.

        Returns:
            data (Dict[str, Union[np.ndarray, float]]): unpacked element.
        """
        # NOTE: in ase databases indexing starts with 1.
        # NOTE: make np arrays writeable
        atoms_row = self._db[idx + 1]
        data = {
            "z": atoms_row.numbers.copy(),
            "y": np.asarray(atoms_row.data["energy"][0]),
            "pos": atoms_row.positions.copy(),
            "forces": atoms_row.data["forces"].copy(),
        }
        return data

    def __len__(self) -> int:
        """Returns number of rows in database."""
        return len(self._db)

    def write(self, values: Dict[str, np.ndarray]) -> None:
        # forces and energy stored in data field
        data_field = {"energy": values.pop("energy"), "forces": values.pop("forces")}
        atoms = Atoms(data=data_field, **values)
        self._db.write(atoms)

    @property
    def metadata(self) -> Dict[str, str]:
        return str(self.metadata)

    @metadata.setter
    def metadata(self, metadata: Dict[str, str]) -> None:
        self.metadata = metadata


class SQLite3Database:
    """Database interface for sqlite3 databases.

    Wraps sqlite3 database with square-shaped data, like Hamiltonian and Overlap matrices
    for model training.

    Args:
        filepath (pathlib.Path): path to existing database or path for new database.
        metadata (Optional[DatasetCard]): dataset metadata.
    """

    type = ".db"

    def __init__(self, filepath: Union[pathlib.Path, str], metadata: Optional[DatasetCard] = None) -> None:
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        if filepath.suffix != self.type:
            raise ValueError(f"Invalid file type: {filepath.suffix}")
        self.filepath = filepath.absolute()
        self._connections = {}

        # parse sample schema and metadata
        if metadata is not None:
            self.name = metadata.name
            self.metadata = metadata.metadata
            self.key_map = _parse_schema(metadata.keys_map)
            self.dtypes = _parse_dtypes(metadata.data_dtypes)
            self.shapes = _parse_shapes(metadata.data_shapes)

    def __getitem__(self, idx: int) -> Dict:
        # SELECT keys FROM data WHERE id=[]
        pass

    def __getitems__(self, idx: Union[List[int], slice]) -> Dict:
        pass

    def __setitem__(self, idx: Union[List[int], slice], values: Union[List[int], slice]) -> Dict:
        pass

    def __delitem__(self, idx: Union[List[int], slice]) -> None:
        pass

    def __len__(self) -> int:
        pass

    def _insert(self, idx: int, data: Dict) -> None:
        pass

    def _insert_many(self, idx: int, data: Dict) -> None:
        pass

    def _get_connection(self) -> None:
        pass

    @property
    def metadata(self) -> Dict[str, str]:
        return str(self.metadata)

    @metadata.setter
    def metadata(self, metadata: Dict[str, str]) -> None:
        self.metadata = metadata


def _blob(array: np.ndarray, dtype: np.dtype) -> memoryview:
    """Convert numpy array to buffer object.

    Args:
        array (np.ndarray): array to convert.
        dtype (np.dtype): array's dtype to save.

    Returns:
        memoryview: buffer object.
    """
    if array is None:
        return None
    if array.dtype == dtype:
        array = array.astype(dtype)
    if not np.little_endian:
        array = array.byteswap()
    return memoryview(np.ascontiguousarray(array))


def _deblob(buf: memoryview, dtype: Optional[np.dtype] = np.float32, shape: Optional[Tuple] = None) -> np.ndarray:
    """Convert buffer object to numpy array.

    Args:
        buf (memoryview): buffer object to convert.
        dtype (np.dtype, optional): dtype of array. Default is np.float32.
        shape (tuple, optional): shape of array. Default is None.

    Returns:
        np.ndarray: numpy array with data.
    """
    if buf is None:
        return np.zeros(shape)
    array = np.frombuffer(buf, dtype)
    if not np.little_endian:
        array = array.byteswap()
    array.shape = shape
    return array


def _parse_schema(mapping: Dict[str, str]) -> Tuple[List]:
    """Returns parsed mapping from datasource keys to sample keys."""
    sample_keys, tables, columns = [], [], []
    for key, value in mapping.items():
        sample_keys.append(key)
        table, column = value.split(".")
        tables.append(table)
        columns.append(column)
    return sample_keys, tables, columns


def _parse_dtypes(dtypes: Dict[str, str]) -> Dict[str, np.dtype]:
    dtypes = {}
    for key, value in dtypes.items():
        dtypes[key] = np.dtype(value)
    return dtypes


def _parse_shapes(shapes: Dict[str, str]) -> Dict[str, tuple]:
    shapes = {}
    for key, value in shapes.items():
        shapes[key] = literal_eval(value)
    return shapes


def _matrix_from_bytes(buf: memoryview, dtype: np.dtype) -> np.ndarray:
    """Helper function creates numpy square matrix from bytes.

    Shape of matrix infered from buffer size and dtype.

    Args:
        buf (memoryview): buffer object to convert.
        dtype (np.dtype): dtype of array.
    """
    elem_num = math.sqrt(len(buf) / dtype.itemsize)
    return _deblob(buf=buf, dtype=dtype, shape=(elem_num, elem_num))
