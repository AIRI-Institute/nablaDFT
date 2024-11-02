import math
import multiprocessing
import os
from enum import Enum
from functools import cached_property
from typing import Dict, List, Union

import apsw  # way faster than sqlite3
import numpy as np


class apswFlags(Enum):
    pass


class SQLDatabase:
    def __init__(self, filename: str, data_schema: Dict, flags=apsw.SQLITE_OPEN_READONLY) -> None:
        pass

    def __getitem__(self, idx: int) -> Dict:
        pass

    def __getitems__(self, idx: Union[List[int], slice]) -> Dict:
        pass

    def __setitem__(self, idx: Union[List[int], slice]) -> Dict:
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
    def metadata(self) -> Dict:
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute("""SELECT * FROM metadata WHERE id=0""").fetchall()
        return data
