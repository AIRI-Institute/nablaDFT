import os

import pytest


@pytest.fixture(autouse=True)
def set_cublas_var():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
