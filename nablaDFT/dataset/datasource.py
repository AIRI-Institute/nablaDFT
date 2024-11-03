"""Module describes interfaces for datasources.

Currently we use sqlite3 databases for energy, hamiltonian and overlap data.

"""

import math
import multiprocessing
import os
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Union

import apsw  # way faster than sqlite3
import ase
import numpy as np
from ase import Atoms


class apswFlags(Enum):
    pass


class EnergyDatabase:
    """Database interface for energy data.

    Wraps the
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, float]]:
        pass

    def __getitems__(self, idx: Union[List[int], int]) -> List[Dict[str, Union[np.ndarray, float]]]:
        pass

    def __setitem__(self, idx: Union[List[int], int], value: Dict[str, Union[np.ndarray, float]]) -> None:
        pass

    def __delitem__(self, idx: Union[List[int], int]) -> None:
        pass

    def _get_connection(self) -> ase.db.sqlite.SQLite3Database:
        conn = ase.db.connect(self.filepath, type="db")
        return conn

    @cached_property
    def metadata(self) -> Dict[str, str]:
        return self._get_connection().metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, str]) -> None:
        self._get_connection().metadata = metadata


class SQLDatabase:
    def __init__(self, filename: str, data_schema: Dict, flags=apsw.SQLITE_OPEN_READONLY) -> None:
        pass

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

    @metadata.setter
    def metadata(self, metadata: Dict[str, str]) -> None:
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READWRITE).cursor()
        cursor.execute("""UPDATE metadata SET value=? WHERE id=0""", (metadata,))
