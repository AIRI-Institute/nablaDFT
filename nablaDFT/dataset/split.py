"""Overrides split strategies from schnetpack.data.splitting"""

import math
from typing import List

import torch
from schnetpack.data.splitting import SplittingStrategy


def abs_split_sizes(dsize, split_sizes: List):
    """Calculates dataset split sizes according to given split ratios"""
    psum = 0
    for idx, size in enumerate(split_sizes):
        if size == 0:
            continue
        elif size <= 1:
            split_sizes[idx] = int(math.floor(size * dsize))
        psum += split_sizes[idx]
    remaining = psum - dsize
    if remaining > 0:
        max_idx = split_sizes.index(max(split_sizes))
        split_sizes[max_idx] += remaining
    return split_sizes


class RandomSplit(SplittingStrategy):

    def split(self, dataset, *split_sizes):
        dsize = len(dataset)
        split_sizes = abs_split_sizes(dsize, list(split_sizes))
        offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
        indices = torch.randperm(sum(split_sizes)).tolist()
        partition_sizes_idx = [
            indices[offset - length : offset]
            for offset, length in zip(offsets, split_sizes)
        ]
        return partition_sizes_idx


class TestSplit(SplittingStrategy):
    """Splitting strategy that allows to put all dataset
    elements in test split without index permutations.
    Used for schnetpack datasets to overcome limitation
    when train and val split are empty.
    """

    def split(self, dataset, *split_sizes):
        dsize = len(dataset)
        partition_sizes_idx = [[], [], list(range(dsize))]  # train, val, test
        return partition_sizes_idx
