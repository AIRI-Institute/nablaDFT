from pathlib import Path

import yaml
import pytest


import nablaDFT


def model_configs():
    model_cfgs = []
    names = []
    model_cfg_path = (Path(nablaDFT.__path__[0]) / "../config/model/").glob("*")
    for cfg_path in model_cfg_path:
        with open(cfg_path, "r") as fin:
            cfg = yaml.safe_load(fin)
        names.append(cfg["model_name"])
        model_cfgs.append(cfg)
    return [pytest.param(cfg, id=name) for cfg, name in zip(model_cfgs, names)]
