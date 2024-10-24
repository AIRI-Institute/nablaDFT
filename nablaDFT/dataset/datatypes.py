from functools import singledispatchmethod
from typing import Protocol


class DataType(Protocol):
    def __init__(self):
        pass
