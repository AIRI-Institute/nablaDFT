from typing import List, Mapping


def _check_ds_len(datasources: List[Mapping]) -> None:
    """Checks that all datasources have the same length.

    Args:
        datasources (List[Mapping]): datasources to check.
    """
    ds_len = [len(datasource) for datasource in datasources]
    if ds_len.count(ds_len[0]) != len(ds_len):
        raise ValueError("Datasources must have the same length")
