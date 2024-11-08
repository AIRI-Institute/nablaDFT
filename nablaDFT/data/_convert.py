import math
from typing import Optional, Tuple

import numpy as np

# Maps database key to numpy array default shape
_default_shapes = {"R": (-1, 3), "F": (-1, 3)}
_default_dtypes = {"Z": np.int32, "E": np.float32, "F": np.float32, "R": np.float32}


def _np_to_blob(array: np.ndarray, dtype: np.dtype) -> memoryview:
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


def _matrix_from_bytes(buf: bytes, dtype: np.dtype, **kwargs) -> np.ndarray:
    """Helper function creates numpy square matrix from bytes.

    Shape of matrix infered from buffer size and dtype.

    Args:
        buf (memoryview): buffer object to convert.
        dtype (np.dtype): dtype of array.
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
    array = np.frombuffer(buf, dtype)
    if not np.little_endian:
        array = array.byteswap()
    if shape is not None:
        array.shape = shape
    else:
        array.shape = (array.size,)
    return array


# Mapping with keys that must be converted in special way.
from_buf_map = {"H": _matrix_from_bytes, "S": _matrix_from_bytes, "C": _matrix_from_bytes}


def np_from_buf(buf: bytes, key: str, **kwargs) -> np.ndarray:
    # skip floats and integers
    if not isinstance(buf, bytes):
        return buf
    if from_buf_map.get(key):
        return from_buf_map[key](buf, **kwargs)
    # use default shape or from kwargs
    shape = kwargs.pop("shape", None)
    dtype = kwargs.pop("dtype", np.float32)
    if shape is None:
        shape = _default_shapes.get(key)
    return _blob_to_np(buf, shape=shape, dtype=dtype, **kwargs)
