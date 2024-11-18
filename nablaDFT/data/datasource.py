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
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

import apsw  # way faster than sqlite3
import ase
import numpy as np
from ase import Atoms

from ._convert import np_from_bytes, np_to_bytes
from ._metadata import DatasourceCard
from .utils import slice_to_list

logger = logging.getLogger(__name__)


sqlite3_datatypes_map = {float: "REAL", int: "INTEGER", str: "TEXT"}


class EnergyDatabase:
    """Database interface for energy data.

    Wraps the ASE sqlite3 database for training NNPs.
    For historical reasons we use ASE databases for energy datasets.

    Args:
        filepath (pathlib.Path): path to existing database or path for new database.
        metadata (Optional[DatasetCard]): optional, dataset metadata.
    """

    type = "db"  # only sqlite3 databases
    _keys_map = {"z": "z", "pos": "pos", "forces": "forces", "y": "y"}

    def __init__(self, filepath: Union[pathlib.Path, str], metadata: Optional[DatasourceCard] = None) -> None:
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
    """Database interface for sqlite3 databases.

    Wraps sqlite3 database, retrieves elements from `data` table.

    Args:
        filepath (pathlib.Path): path to existing database or path for new database.
        metadata (Optional[DatasetCard]): dataset metadata.
    """

    type = ".db"

    def __init__(self, filepath: Union[pathlib.Path, str], metadata: Optional[DatasourceCard] = None) -> None:
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        if filepath.suffix != self.type:
            raise ValueError(f"Invalid file type: {filepath.suffix}")
        self.filepath = filepath.absolute()
        # initialize metadata
        self.desc: Dict[str, str] = None
        self.metadata: Dict[str, Any] = None
        self.columns: List[str] = []
        self._keys_map: Dict[str, str] = {}
        self._dtypes: Dict[str, np.dtype] = {}
        self._shapes: Dict[str, tuple] = {}

        self._connections = {}

        if not self.filepath.exists():
            self._create(filepath, metadata)
        else:
            # check if `data` table exists
            if "data" not in self._get_tables_list():
                raise ValueError("Table `data` not found in database")
            # parse sample schema and metadata
            self._parse_metadata(metadata)
        logger.info(f"Created new database: {filepath}")

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
            data = self._unpack(cursor.execute(query).fetchone())
            # rename keys if needed
            if self._keys_map != self.columns:
                data = {new_key: data[old_key] for new_key, old_key in self._keys_map.items()}
            return data
        else:
            return self._get_many(idx)

    def write(self, data: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]):
        self._check_data_keys(data)
        if isinstance(data, dict):
            self._insert(data)
        elif isinstance(data, list):
            self._insert_many(data)
        else:
            raise TypeError(f"Expected Dict or List[Dict], got {type(data)}")

    def update(
        self, data: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]], idx: Union[int, slice, List[int]]
    ):
        if isinstance(data, dict):
            data = [data]
        self._check_data_keys(data)
        self._update(data, idx)

    def delete(self, idx: Union[int, slice, List]):
        if isinstance(idx, slice):
            idx = slice_to_list(idx)
        query = self._construct_delete(idx)
        with self._get_connection() as conn:
            conn.execute(query)

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
        data = [self._unpack(chunk) for chunk in cursor.execute(query).fetchall()]
        # rename keys if needed
        if self._keys_map != self.columns:
            data = [
                {new_key: data_chunk[old_key] for new_key, old_key in self._keys_map.items()} for data_chunk in data
            ]
        return data

    def __len__(self) -> int:
        """Returns number of rows in database."""
        return self._table_len("data")

    def _create(self, filepath: pathlib.Path, metadata: DatasourceCard):
        if metadata is None:
            raise ValueError(f"Can't create table {filepath} without metadata.")
        if not metadata.columns:
            raise ValueError("Column names not specified.")
        if not metadata._dtypes:
            raise ValueError("Data types not specified.")
        if not metadata._shapes:
            raise ValueError("Data shapes not specified.")
        self._create_table("data", metadata._dtypes, metadata.columns)
        self._parse_metadata(metadata)

    def _get_connection(self) -> apsw.Connection:
        key = multiprocessing.current_process().name
        if key not in self._connections.keys():
            self._connections[key] = apsw.Connection(str(self.filepath))
            self._connections[key].setbusytimeout(300000)  # 5 minute timeout
        return self._connections[key]

    def _unpack(self, data: Tuple[bytes]) -> Dict[str, np.ndarray]:
        """Unpacks data from sqlite3 database."""
        data_dict = {}
        for idx, col in enumerate(self.columns):
            dtype = self._dtypes.get(col, None)
            shape = self._shapes.get(col, None)
            data_dict[col] = np_from_bytes(
                data[idx],
                col,
                dtype=dtype,
                shape=shape,
            )
        return data_dict

    def _pack(self, data: Dict[str, np.ndarray]) -> Dict[str, bytes]:
        bytes_dict = {}
        for key in data.keys():
            dtype = self._dtypes.get(key, None)
            bytes_dict[key] = np_to_bytes(data[key], dtype=dtype)
        return bytes_dict

    def _insert(self, data: Dict[str, np.ndarray], table: str = "data") -> None:
        data = self._pack(data)
        query = self._construct_insert(table)
        idx: Tuple = (len(self),)
        # take data in columns order and add id
        data: Tuple = idx + itemgetter(*self.columns)(data)
        # write and commit
        with self._get_connection() as conn:
            conn.execute(query, data)

    def _insert_many(
        self,
        data: List[Dict[str, np.ndarray]],
        table_name: str = "data",
    ) -> None:
        data = [self._pack(sample) for sample in data]
        query = self._construct_insert(table_name)
        data_len = len(data)
        table_len = self._table_len(table_name)
        idx = tuple(range(table_len, table_len + data_len))
        # take data in columns order and add id
        data: Tuple[Tuple] = tuple([(idx[i],) + itemgetter(*self.columns)(sample) for i, sample in enumerate(data)])
        with self._get_connection() as conn:
            conn.executemany(query, data)

    def _update(
        self, data: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]], idx: Union[int, slice, List]
    ) -> None:
        if isinstance(idx, slice):
            idx = slice_to_list(idx)
        # check data keys consistency
        self._create_table("temp", self._dtypes, self.columns)
        query = self._construct_insert("temp")
        if isinstance(idx, int):
            idx = (idx,)
        else:
            idx = tuple(idx)
        data = [self._pack(sample) for sample in data]
        data: Tuple[Tuple] = tuple([(idx[i],) + itemgetter(*self.columns)(sample) for i, sample in enumerate(data)])
        cols_map = ",\n".join([f"{col} = temp.{col}" for col in self.columns])
        with self._get_connection() as conn:
            conn.executemany(query, data)
            conn.execute(f"""UPDATE data\nSET\n{cols_map}\nFROM temp WHERE data.id == temp.id""")
            conn.execute("""DROP TABLE temp""")

    def _get_table_schema(self, table_name: str) -> List:
        """Returns table schema from sqlite3 database."""
        columns = []
        conn = self._get_connection()
        schema = conn.pragma(f"table_info({table_name})")
        for entry in schema:
            # TODO: delete this when we delete core hamiltonian column
            if entry[1] in ["id", "C"]:
                continue
            columns.append(entry[1])
        return columns

    def _get_tables_list(self) -> List[str]:
        cursor = self._get_connection().cursor()
        tables_list = [info[1] for info in cursor.execute("SELECT * FROM sqlite_master WHERE type='table'").fetchall()]
        return tables_list

    def _construct_select(self, idx: Union[int, List[int], slice]) -> str:
        cols = ", ".join(self.columns)
        if isinstance(idx, int):
            return f"SELECT {cols} FROM data WHERE id={idx}"
        else:
            return f"SELECT {cols} FROM data WHERE id IN ({str(idx)[1:-1]})"

    def _construct_delete(self, idx: Union[int, List[int], slice]) -> str:
        if isinstance(idx, int):
            return f"DELETE FROM data WHERE id={idx}"
        else:
            return f"DELETE FROM data WHERE id IN ({str(idx)[1:-1]})"

    def _construct_insert(self, table_name: str):
        cols = ", ".join(self.columns)
        query = f"""INSERT INTO {table_name} (id, {cols})
                    VALUES ({", ".join(["?" for _ in range(len(self.columns) + 1)])})
                """
        return query

    def _create_table(self, name: str, dtypes: Dict[str, str], columns: List[str]):
        col_types = [self._to_sql_type(data_type) for data_type in dtypes]
        table_schema = ",\n".join([f" {col_name} {col_type}" for col_name, col_type in zip(columns, col_types)])
        query = f"CREATE TABLE IF NOT EXISTS {name} \n (id INTEGER NOT NULL PRIMARY KEY,\n {table_schema})"
        with self._get_connection() as conn:
            conn.execute(query)

    def _check_data_keys(self, data: List[Dict[str, np.ndarray]]):
        if not isinstance(data, list):
            data = [data]
        for i, sample in enumerate(data):
            if [*sample.keys()] != self.columns:
                no_keys = set(self.columns) - set(sample.keys())
                raise ValueError(f"No key in data at index {i}: {no_keys}")

    def _to_sql_type(self, data_type: str):
        sqlite3_datatype = sqlite3_datatypes_map.get(data_type, None)
        if sqlite3_datatype is None:
            sqlite3_datatype = "BLOB"
        return sqlite3_datatype

    def _parse_metadata(self, metadata: DatasourceCard) -> None:
        if metadata is None:
            # use full table `data`
            self.columns = self._get_table_schema("data")
            self._keys_map = {key: key for key in self.columns}
        else:
            self.desc = metadata.desc
            self.metadata = metadata.metadata
            self.columns = metadata.columns
            if metadata._keys_map:
                self._keys_map = metadata._keys_map
            else:
                self._keys_map = {key: key for key in self.columns}
            self._dtypes = metadata._dtypes
            self._shapes = metadata._shapes

    def _table_len(self, table_name: str):
        cursor = self._get_connection().cursor()
        return cursor.execute(f"""SELECT COUNT(*) FROM {table_name}""").fetchone()[0]

    @property
    def units(self):
        units = self.metadata.metadata.get("units", None)
        return units

    @property
    def method(self):
        method = self.metadata.metadata.get("method", None)
        return method


Datasource = Union[EnergyDatabase, SQLite3Database]
