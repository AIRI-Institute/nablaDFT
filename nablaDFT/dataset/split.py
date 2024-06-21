"""Overrides split strategies from schnetpack.data.splitting"""

from typing import List

from schnetpack.data import ASEAtomsData
from schnetpack.data.splitting import SplittingStrategy


class TestSplit(SplittingStrategy):
    """Splitting strategy that allows to put all dataset elements in test split without index permutations.
    Used for schnetpack datasets to overcome limitation when train and val split are empty.
    """

    def split(self, dataset: ASEAtomsData, *split_sizes: List):
        """Returns test split indexes, leaves train and val indices empty.

        Args:
            dataset (ASEAtomsData): test dataset.
            split_sizes (List): remains for consistency with other SplittingStrategy classes.
        """
        dsize = len(dataset)
        partition_sizes_idx = [[], [], list(range(dsize))]  # train, val, test
        return partition_sizes_idx
