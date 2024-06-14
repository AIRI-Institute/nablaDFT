import importlib

import pytest

from .configs import model_configs


@pytest.mark.model
@pytest.mark.parametrize("model_cfg", model_configs())
def test_torch_model(model_cfg, dataset_pyg):
    torch_model_cfg = model_cfg['model']
    target = torch_model_cfg['_target_']
    del torch_model_cfg['_target_']
    model_cls_str = target.split(".")[-1]
    module = ".".join(target.split(".")[:-1])
    model_cls = getattr(importlib.import_module(module), model_cls_str)
    model = model_cls(**torch_model_cfg)
    assert model
