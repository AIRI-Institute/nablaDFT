from pathlib import Path

import yaml
import pytest


import nablaDFT


def model_configs():
    model_cfgs = []
    model_cfg_path = (Path(nablaDFT.__path__[0]) / "../config/model/").glob("*")
    for cfg_path in model_cfg_path:
        if "painn" not in cfg_path.name or "schnet" not in cfg_path.name:
            with open(cfg_path, "r") as fin:
                cfg = pytest.param(
                    yaml.safe_load(fin)
                )
            model_cfgs.append(cfg)
    return model_cfgs
