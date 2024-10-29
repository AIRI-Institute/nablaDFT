"""Collate functions for combining samples into batches."""

from typing import Callable, Dict, Optional, Tuple, Type, Union


def collate_hamiltonian(
    batch, *, default_collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None
):
    pass
