"""Module describes interfaces for datasources.

Currently we use sqlite3 databases for energy, hamiltonian and overlap data.

"""

import pathlib
from functools import cached_property
from typing import Dict, List, Optional, Union

import apsw  # way faster than sqlite3
import ase
import numpy as np
from ase import Atoms

from ._metadata import DatasetMetadata


class EnergyDatabase:
    """Database interface for energy data.

    Wraps the ASE sqlite3 database for training NNPs.

    .. NOTE: indexing starts with 1.

    Args:
        filepath (pathlib.Path): path to existing database or path for new database.
        mapping (Dict[str, str]): mapping from column names in database to sample's keys.
    """

    type = "db"  # only sqlite3 databases

    def __init__(self, filepath: pathlib.Path, metadata: Optional[DatasetMetadata] = None) -> None:
        self.filepath = filepath
        self._metadata = metadata

        self._db: ase.db.sqlite.SQLite3Database = ase.db.connect(self.filepath, type=self.type)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, float]]:
        """Returns unpacked element from ASE database.

        Args:
            idx (int): index of row to return.

        Returns:
            data (Dict[str, Union[np.ndarray, float]]): unpacked element.
        """
        atoms_row = self._db[idx]
        data = {
            "z": atoms_row.numbers,
            "y": atoms_row.data["energy"][0],
            "pos": atoms_row.positions,
            "forces": atoms_row.data["forces"],
        }
        return data

    def write(self, atoms: Atoms) -> None:
        self._db.write(atoms)


class SQLDatabase:
    """Database interface for sqlite3 databases.

    Wraps sqlite3 database for training Hamiltonian-like models.

    Args:
        filepath (pathlib.Path): path to existing database or path for new database.
    """

    def __init__(self, filepath: pathlib.Path, data_schema: Dict, flags=apsw.SQLITE_OPEN_READONLY) -> None:
        self.filepath = filepath

    def __getitem__(self, idx: int) -> Dict:
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

    @cached_property
    def metadata(self) -> Dict[str, str]:
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute("""SELECT * FROM metadata WHERE id=0""").fetchall()
        return data


class CSVFile:
    def __init__(self):
        pass

    def __getitem__(self, idx: int):
        pass

    def __getitems__(self, idx: Union[List[int], int]):
        pass

    def __setitem__(self, idx: Union[List[int], int], value: Dict[str, Union[np.ndarray, float]]):
        pass

    def __delitem__(self, idx: Union[List[int], int]):
        pass

    def __len__(self) -> int:
        pass

    @cached_property
    def metadata(self) -> Dict[str, str]:
        pass


def _blob(array: np.ndarray) -> memoryview:
    """Convert numpy array to buffer object.

    Args:
        array (np.ndarray): array to convert.

    Returns:
        memoryview: buffer object.
    """
    if array is None:
        return None
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    if array.dtype == np.int64:
        array = array.astype(np.int32)
    if not np.little_endian:
        array = array.byteswap()
    return memoryview(np.ascontiguousarray(array))


def _deblob(buf: memoryview, dtype=np.float32, shape=None) -> np.ndarray:
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
