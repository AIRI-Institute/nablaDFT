import operator
from functools import reduce
from typing import Dict, List, Mapping


def check_ds_len(datasources: List[Mapping]) -> None:
    """Checks that all datasources have the same length.

    Args:
        datasources (List[Mapping]): datasources to check.
    """
    ds_len = [len(datasource) for datasource in datasources]
    if ds_len.count(ds_len[0]) != len(ds_len):
        raise ValueError("Datasources must have the same length")


def slice_to_list(slice_: slice) -> List[int]:
    """Converts slice object to list."""
    step = slice_.step if slice_.step else 1
    start = slice_.start if slice_.start else 0
    return list(range(start, slice_.stop, step))


def _merge_dicts(dicts: List[Dict]):
    return reduce(operator.ior, dicts, {})


def merge_samples(samples: List[List[Dict]]):
    """Merges samples list retrieved from different datasources in one list."""
    it = zip(*samples)
    return [_merge_dicts(sample) for sample in it]
