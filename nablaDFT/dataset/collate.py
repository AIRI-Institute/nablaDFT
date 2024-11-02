"""Collate functions for combining samples into batches."""

# TODO: current version will fail with torch.Dataset because it will try to collate squared matrices.

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import default_convert
from torch_geometric.data import Batch

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


def collate_pyg(
    batch: List[Dict[str, np.ndarray]],
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    """Collates list of dicts with numpy arrays into torch.geometric.data.Batch object.

    Supports data attributes with different squared shape.
    Function signature is the same as torch.utils.data.dataloader.default_collate.

    Args:
        batch (List[Dict[str, np.ndarray]]): a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Returns:
        pyg_batch (torch_geometric.data.Batch): collated batch.
    """
    # use default collate
    pyg_batch = Batch.from_data_list(batch, exclude_keys=SQUARE_SHAPED_KEYS)
    # collate square shaped elements and add to batch
    pyg_batch.update(__collect_square_shaped_data(batch))
    return pyg_batch


def __collect_square_shaped_data(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Collates square shaped data from a batch of data.

    Args:
        batch (List[Dict[str, np.ndarray]]): a batch with square shaped data.

    Returns:
        square_shaped_data (Dict[str, torch.Tensor]): collected data.
    """
    elem = batch[0]
    square_shaped_data = {}
    key_present = [key in elem.keys() for key in SQUARE_SHAPED_KEYS]
    if any(key_present):
        for idx, key in enumerate(SQUARE_SHAPED_KEYS):
            if key_present[idx]:
                square_shaped_data[key] = [
                    torch.from_numpy(batch[i][key]) for i in range(len(batch))
                ]  # or use default_convert ?
    return square_shaped_data
