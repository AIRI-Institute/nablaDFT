from pathlib import Path

import yaml

import nablaDFT


def read_model_cfg(name: str):
    model_cfg_path = Path(nablaDFT.__path__[0]) / f"../config/model/{name}.yaml"
    with open(model_cfg_path, "r") as fin:
        cfg = yaml.safe_load(fin)
    return cfg
