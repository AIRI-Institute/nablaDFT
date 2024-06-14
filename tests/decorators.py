from typing import Callable

import torch


def withCUDA(func: Callable) -> Callable:
    r"""A decorator to test both on CPU and CUDA (if available).

    Borrowed from https://github.com/pyg-team/pytorch_geometric/blob/0419f0f2d372056b0d7b939e15069b41b35a98a7/torch_geometric/testing/decorators.py
    """
    import pytest

    devices = [pytest.param(torch.device("cpu"), id="cpu")]
    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device("cuda:0"), id="cuda:0"))

    return pytest.mark.parametrize("device", devices)(func)
