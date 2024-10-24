# TODO: write examples
"""Module for interacting with various data sources.

This module provides a generic interface for working with different data sources,
such as SQL or LMDB databases. It defines a `DataSource` protocol that
can be implemented by concrete data source classes.

The goal of this module is to provide a unified way of accessing and manipulating
data from different sources, allowing for polymorphic behavior and decoupling
the application logic from the specific data source implementation.

Open existing SQL database with Hamiltonian data:
.. code-block:: python
    from nablaDFT.dataset.database import (
        SQLDataSource,
    )

    # Create a SQL data source
    sql_ds = SQLDataSource(
        "sqlite:///example.db"
    )

Open ASE database with molecules:
.. code-block:: python
    from nablaDFT.dataset.database import (
        SQLDataSource,
    )

    # Create a SQL data source
    sql_ds = SQLDataSource(
        "ase", "example.db"
    )

"""

from abc import abstractmethod
from typing import Any, Protocol, TypeVar

from .datatypes import DataType

T = TypeVar("T")


class DataSource(Protocol[T]):
    """Base protocol for data sources like databases or files.

    Provides a generic interface for working with different data sources.
    Concrete implementations must subclass this class and implement the
    abstract methods.

    Attributes:
        datatype (DataType): the data type of the data in the source, must be one of defined in datatypes.py.
        path (str): the path to the data source.
    """

    datatype: DataType
    path: str

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        pass

    @abstractmethod
    def __setitem__(self, idx: int, item: T) -> None:
        pass

    @abstractmethod
    def __delitem__(self, idx: int) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class SQLDataSource:
    """ """

    def __init__(self, datatype: DataType, path: str) -> None:
        self.datatype = datatype
        self.path = path

    def __getitem__(self, idx: int) -> T:
        pass

    def __setitem__(self, idx: int, item: T) -> None:
        pass

    def __delitem__(self, idx: int) -> None:
        pass

    def __len__(self) -> int:
        pass

    def get(self, idx: int) -> T:
        pass

    def getmany(self, idxs: slice) -> Tuple[T]:
        pass
    