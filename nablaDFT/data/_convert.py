import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import default_convert
from torch_geometric.data import Data

SampleType = Union[np.ndarray, torch.Tensor]

# Maps nablaDFT database key to numpy array default shape
_default_shapes = {"R": (-1, 3), "F": (-1, 3)}
# Maps nablaDFT database key to numpy array default dtype
_default_dtypes = {
    "Z": np.dtype("int32"),  # atoms numbers
    "E": np.dtype("float32"),  # energy
    "F": np.dtype("float32"),  # atomic forces
    "R": np.dtype("float32"),  # atom coordinates
    "H": np.dtype("float32"),  # hamiltonian matrix
    "S": np.dtype("float32"),  # overlap matrix
    "C": np.dtype("float32"),  # core hamiltonian matrix
}


def np_to_bytes(array: np.ndarray, dtype: np.dtype) -> bytes:
    """Convert numpy array to buffer object.

    Args:
        array (np.ndarray): array to convert.
        dtype (np.dtype): array's dtype to save.

    Returns:
        memoryview: buffer object.
    """
    if array is None:
        return None
    # explicitly check that elements class is the same as desired dtype
    if array.dtype.type != dtype:
        array = array.astype(dtype)
    if not np.little_endian:
        array = array.byteswap()
    return array.tobytes()


def _matrix_from_bytes(buf: bytes, dtype: np.dtype, **kwargs) -> np.ndarray:
    """Helper function creates numpy square matrix from bytes.

    Shape of matrix infered from buffer size and dtype.

    Args:
        buf (memoryview): buffer object to convert.
        dtype (np.dtype): dtype of array.
        kwargs (Dict[Any]): optional key-word args.
    """
    elem_num = int(math.sqrt(len(buf) / dtype.itemsize))
    return _blob_to_np(buf=buf, dtype=dtype, shape=(elem_num, elem_num))


def _blob_to_np(buf: bytes, dtype: Optional[np.dtype] = np.float32, shape: Optional[Tuple] = None) -> np.ndarray:
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
    array = np.frombuffer(buf, dtype).copy()
    if not np.little_endian:
        array = array.byteswap()
    if shape is not None:
        array.shape = shape
    else:
        array.shape = (array.size,)
    array.setflags(write=1)
    return array


# Mapping with keys that must be converted in special way.
from_buf_map = {"H": _matrix_from_bytes, "S": _matrix_from_bytes, "C": _matrix_from_bytes}


def np_from_bytes(buf: bytes, key: str, **kwargs) -> np.ndarray:
    shape = kwargs.pop("shape", None)
    dtype = kwargs.pop("dtype", np.float32)
    if shape is None:
        shape = _default_shapes.get(key)
    if dtype is None:
        dtype = _default_dtypes.get(key)
    # floats and integers must be converted to 0-dimensional np.arrays
    if isinstance(buf, int) or isinstance(buf, float) or isinstance(buf, str):
        return np.array(buf, dtype=dtype).copy()
    # use default shape or from kwargs
    if from_buf_map.get(key):
        return from_buf_map[key](buf, dtype=dtype, **kwargs)
    arr = _blob_to_np(buf, shape=shape, dtype=dtype, **kwargs)
    return arr


def to_pyg_data(sample: Union[List[Dict[str, SampleType]], Dict[str, SampleType]]) -> Union[List[Data], Data]:
    """Convert single or list of samples to torch.geometric.data.Data object."""
    if isinstance(sample, List):
        return [Data(**default_convert(data)) for data in sample]
    else:
        return Data(**default_convert(sample))
