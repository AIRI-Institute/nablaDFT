import operator
from functools import reduce
from typing import Dict, List

_pyg_num_node_keys = {"num_nodes", "x", "pos", "batch", "adj", "adj_t", "edge_index", "face"}


def check_ds_len(datasources: List) -> None:
    """Checks that all datasources have the same length.

    Args:
        datasources (List[Datasource]): datasources to check.
    """
    ds_len = [len(datasource) for datasource in datasources]
    if ds_len.count(ds_len[0]) != len(ds_len):
        raise ValueError("Datasources must have the same length")


def check_ds_keys_map(datasources: List):
    """Checks that all datasources have key mapping for PyG batching.

    Args:
        datasources (List[Datasource]): datasources to check.
    """
    for datasource in datasources:
        if not datasource._keys_map:
            raise AttributeError(
                f"""Datasource {datasource.__class__}(filepath={datasource.filepath})\
                  doesn't have _keys_map param, which required for PyGDataset"""
            )
        else:
            keys = set(datasource._keys_map.keys())
            if not len(keys & _pyg_num_node_keys) > 0 and not len([key for key in keys if "node" in key]) > 0:
                raise AttributeError(
                    f"""Datasource {datasource.__class__}(filepath={datasource.filepath})\
                  doesn't have keys for batching. Consider add key with name {_pyg_num_node_keys}\
                  or key with "node" in its name"""
                )


def slice_to_list(slice_: slice) -> List[int]:
    """Converts slice object to list."""
    step = slice_.step if slice_.step else 1
    start = slice_.start if slice_.start else 0
    return list(range(start, slice_.stop, step))


def _merge_dicts(dicts: List[Dict]):
    """Merge list of dicts into one dict."""
    return reduce(operator.ior, dicts, {})


def merge_samples(samples: List[List[Dict]]):
    """Merges samples list retrieved from different datasources in one list."""
    it = zip(*samples)
    return [_merge_dicts(sample) for sample in it]
