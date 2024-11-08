"""Collate functions for combining samples into batches."""

# TODO: current version will fail with torch.Dataset because it will try to collate squared matrices.

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import default_convert
from torch_geometric.data import Batch, Data
from torch_geometric.data.collate import collate

SQUARE_SHAPED_KEYS = [
    "H",  # Hamiltonian
    "S",  # Overlap matrix
    "C",  # Core hamiltonian
    "D",  # Electronic density matrix
]
"""Keys with data presented in square matrix form.

Keys will be separated differently by collate functions to preventing
attempts to concatenate (torch_geometric) or stack (torch) square matrices
with different shapes.
"""


def _collate_pyg_batch(
    batch: List[Data],
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    """Collates list of torch_geometric.data.Data objects into torch.geometric.data.Batch object.

    The only purpose of this function: dispatch collate_fn call for pyg.data.Data objects.
    """
    pyg_batch = Batch.from_data_list(batch, exclude_keys=SQUARE_SHAPED_KEYS)
    squared_shape_data, square_slices, square_incs = __collect_square_shaped_data(batch)
    pyg_batch.update(squared_shape_data)
    pyg_batch._slice_dict.update(square_slices)
    pyg_batch._inc_dict.update(square_incs)
    return pyg_batch


def collate_pyg(
    batch: List[Data],
    *,
    increment: bool = False,
    add_batch: bool = False,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    """Collates list of torch_geometric.data.Data objects for pytorch_geometric.data.InMemoryDataset format.

    Supports data attributes with different squared shape.
    Function signature is the same as torch.utils.data.dataloader.default_collate.

    Args:
        batch (List[Data]): a single batch to be collated.
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.
    """
    pyg_batch, slice_dict, inc_dict = collate(
        batch[0].__class__,
        data_list=batch,
        increment=increment,
        add_batch=add_batch,
        exclude_keys=SQUARE_SHAPED_KEYS,
    )
    # collate square-shape elements and add to batch
    squared_shape_data, square_slices, square_incs = __collect_square_shaped_data(batch)
    pyg_batch.update(squared_shape_data)
    slice_dict.update(square_slices)
    inc_dict.update(square_incs)
    return pyg_batch, slice_dict, inc_dict


def __collect_square_shaped_data(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Collects square shaped data from a batch of data.

    Args:
        batch (List[Dict[str, np.ndarray]]): a batch with square shaped data.
        return_slices (bool): whether to return a dictionary of slices in addition to the converted data.

    Returns:
        square_shaped_data (Dict[str, torch.Tensor]): collected data.
    """
    elem = batch[0]
    square_shaped_data = {}
    slices_dict = {}
    inc_dict = {}
    key_present = [key in elem.keys() for key in SQUARE_SHAPED_KEYS]
    if any(key_present):
        for idx, key in enumerate(SQUARE_SHAPED_KEYS):
            if key_present[idx]:
                square_shaped_data[key] = [default_convert(batch[i][key]) for i in range(len(batch))]
                slices_dict[key] = torch.arange(0, len(batch) + 1, dtype=torch.long)
                inc_dict[key] = torch.zeros(len(batch), dtype=torch.long)
    return square_shaped_data, slices_dict, inc_dict
