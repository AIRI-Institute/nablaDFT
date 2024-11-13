"""Module describes interfaces for datasources, used for model training and EDA.

Examples:
--------
.. code-block:: python
    from nablaDFT.data import EnergyDatabase

    >>> datasource = EnergyDatabase("path-to-database")
    >>> datasource[0]
    >>> {
    'z': array([...], dtype=int32),
    'y': array(...),
    'pos': array([...]),
    'forces': array([...]),
    }

Create new DataBase with the same schema as in downloaded datasource:
.. code-block:: python
    >>> from nablaDFT.data import SQLite3Database
    >>> datasource = SQLite3Database("path-to-database")
    >>> metadata = datasource.metadata
    >>> new_datasource = SQLite3Database("path-to-new-database", schema=schema)
    >>> new_datasource[idx] = datasource[0]  # write sample to new database
"""

import logging
import multiprocessing
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import apsw  # way faster than sqlite3
import ase
import numpy as np
from ase import Atoms

from ._convert import np_from_buf
from ._metadata import DatasetCard
from .utils import slice_to_list

logger = logging.getLogger(__name__)


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
            self.desc = metadata.desc
            self.metadata = metadata.metadata

    def __getitem__(self, idx: Union[int, slice, List[int]]) -> Dict[str, np.ndarray]:
        """Returns unpacked element from ASE database.

        Args:
            idx (int): index of row to return.

        Returns:
            data (Dict[str, Union[np.ndarray, float]]): unpacked element.
        """
        # NOTE: in ase databases indexing starts with 1.
        # NOTE: make np arrays writeable
        if isinstance(idx, int):
            atoms_row = self._db[idx + 1]
            data = {
                "z": atoms_row.numbers.copy(),
                "y": np.asarray(atoms_row.data["energy"][0]),
                "pos": atoms_row.positions.copy(),
                "forces": atoms_row.data["forces"].copy(),
            }
        else:
            if isinstance(idx, slice):
                idx = slice_to_list(idx)
            data = [self.__getitem__(i) for i in idx]
        return data

    def __len__(self) -> int:
        """Returns number of rows in database."""
        return len(self._db)

    def write(self, values: Dict[str, np.ndarray]) -> None:
        # forces and energy stored in data field
        data_field = {"energy": values.pop("energy"), "forces": values.pop("forces")}
        atoms = Atoms(data=data_field, **values)
        self._db.write(atoms)


class SQLite3Database:
    """Read-only database interface for sqlite3 databases.

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
        # initialize metadata
        self.desc: Dict[str, str] = None
        self.metadata: Dict[Any, Any] = None
        self._keys_map: Tuple[List[str], List[str], List[str]] = None
        self._dtypes: Dict[str, str] = {}
        self._shapes: Dict[str, str] = {}

        self._connections = {}

        if not self.filepath.exists():
            logger.info(f"Creating new database: {filepath}")
            self._create(filepath, metadata)
            # raise FileNotFoundError(f"Database not found: {self.filepath}")
        else:
            # check if `data` table exists
            if "data" not in self._get_tables_list():
                raise ValueError("Table `data` not found in database")
            # parse sample schema and metadata
            self._parse_metadata(metadata)

    def __getitem__(
        self, idx: Union[int, List[int], slice]
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Returns unpacked element from sqlite3 database.

        Args:
            idx (Union[int, List[int], slice]): row index of row to return.

        Returns:
            data (Dict[str, np.ndarray]): unpacked element.
        """
        if isinstance(idx, int):
            query = self._construct_select(idx)
            cursor = self._get_connection().cursor()
            raw_data = self._unpack(cursor.execute(query).fetchone())
            data = {key: raw_data[self._keys_map[-1][i]] for i, key in enumerate(self._keys_map[0])}
            return data
        else:
            return self._get_many(idx)

    def _get_many(self, idx: Union[List[int], slice]) -> List[Dict[str, np.ndarray]]:
        """Returns unpacked elements from sqlite3 database.

        Args:
            idx (Union[List[int], slice]): indexes of rows to return.

        Returns:
            data (List[Dict[str, np.ndarray]]): list of unpacked elements.
        """
        if isinstance(idx, slice):
            idx = slice_to_list(idx)
        cursor = self._get_connection().cursor()
        query = self._construct_select(idx)
        raw_data = [self._unpack(chunk) for chunk in cursor.execute(query).fetchall()]
        data = [
            {sample_key: data_chunk[db_key] for sample_key, db_key in zip(self._keys_map[0], self._keys_map[-1])}
            for data_chunk in raw_data
        ]
        return data

    def _create(self, filepath: pathlib.Path, metadata: DatasetCard):
        if metadata is None:
            raise ValueError(f"Can't create table {filepath} without metadata")
        cursor = self._get_connection().cursor()
        col_types = [self._to_sql_type(data_type) for data_type in self._dtypes]
        tables, db_keys = self._keys_map[1:]
        sql_schema = {}  # {'table': {'col_name': 'sql_type'}}
        for idx, table in enumerate(tables):
            if sql_schema.get(table):
                sql_schema[table].append({db_keys[idx]: col_types[idx]})
            else:
                sql_schema[table] = [db_keys[idx]]
        for table_name in sql_schema.keys():
            table_schema = ",\n".join(
                [f" {col_name} {col_type}" for col_name, col_type in zip(sql_schema[table].items())]
            )
            query = f"CREATE TABLE IF NOT EXISTS {table_name} \n (id INTEGER NOT NULL PRIMARY KEY,\n {table_schema})"
            cursor.execute(query)

    def __setitem__(self, idx, data: Dict[str, np.ndarray]) -> None:
        pass

    def __delitem__(self, idx) -> None:
        pass

    def __len__(self) -> int:
        """Returns number of rows in database."""
        cursor = self._get_connection().cursor()
        return cursor.execute("""SELECT COUNT(*) FROM data""").fetchone()[0]

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READWRITE) -> apsw.Connection:
        key = multiprocessing.current_process().name
        if key not in self._connections.keys():
            self._connections[key] = apsw.Connection(str(self.filepath), flags=flags)
            self._connections[key].setbusytimeout(300000)  # 5 minute timeout
        return self._connections[key]

    def _unpack(self, data: Tuple[memoryview]) -> Dict[str, np.ndarray]:
        """Unpacks data from sqlite3 database."""
        # retrieve column names
        data_dict = {}
        for idx, key in enumerate(self._keys_map[-1]):
            dtype = self._dtypes.get(f"{self._keys_map[1][idx]}.{self._keys_map[2][idx]}", None)
            shape = self._shapes.get(f"{self._keys_map[1][idx]}.{self._keys_map[2][idx]}", None)
            data_dict[key] = np_from_buf(
                data[idx],
                key,
                dtype=dtype,
                shape=shape,
            )
        return data_dict

    def _pack(self, data: Dict[str, np.ndarray]) -> None:
        pass

    def _insert(self, data) -> None:
        pass

    def _update(self, data, idx) -> None:
        pass

    def _get_table_schema(self, table_name: str) -> List:
        """Returns table schema from sqlite3 database."""
        keys, columns = [], []
        conn = self._get_connection()
        schema = conn.pragma(f"table_info({table_name})")
        for entry in schema:
            # TODO: delete this when we delete core hamiltonian column
            if entry[1] in ["id", "C"]:
                continue
            keys.append(entry[1])
            columns.append(entry[1])
        tables = [table_name for _ in range(len(columns))]
        return [keys, tables, columns]

    def _get_tables_list(self) -> List[str]:
        cursor = self._get_connection().cursor()
        tables_list = [info[1] for info in cursor.execute("SELECT * FROM sqlite_master WHERE type='table'").fetchall()]
        return tables_list

    def _construct_select(self, idx: Union[int, List[int], slice]) -> str:
        _, tables, columns = self._keys_map
        keys = ", ".join(f"{table}.{column}" for table, column in zip(tables, columns))
        tables = ", ".join(f"{table}" for table in set(tables))
        if isinstance(idx, int):
            return f"SELECT {keys} FROM {tables} WHERE id={idx}"
        else:
            return f"SELECT {keys} FROM {tables} WHERE id IN ({str(idx)[1:-1]})"

    def _construct_insert(self):
        pass

    def _construct_update(self):
        pass

    def _to_sql_type(self):  # ???
        pass

    def _parse_metadata(self, metadata: DatasetCard) -> None:
        if metadata is None:
            # use full table `data`
            self._keys_map = self._get_table_schema("data")
        else:
            self.desc = metadata.desc
            self.metadata = metadata.metadata
            self._keys_map = metadata._keys_map
            self._dtypes = metadata._data_dtypes
            self._shapes = metadata._data_shapes
