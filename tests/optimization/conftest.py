from pathlib import Path

import nablaDFT
import pytest
from hydra import compose, initialize_config_dir


@pytest.fixture(
    params=[
        "dimenetplusplus.yaml",
        "equiformer_v2_oc20.yaml",
        "escn-oc.yaml",
        "gemnet-oc.yaml",
        "graphormer3d-small.yaml",
    ]
)
def pyg_optim_config(request, tmp_path):
    cfg_dir_path = str((Path(nablaDFT.__path__[0]) / "../config").resolve())
    with initialize_config_dir(version_base="1.2", config_dir=cfg_dir_path, job_name=f"test_{request.param}"):
        config = compose(config_name="gemnet-oc_optim.yaml", overrides=[f"model={request.param}"])
    model_name = config.model.model_name
    config.pretrained = f"{model_name}_train_large"
    config.input_db = (Path(nablaDFT.__path__[0]) / "../tests/data/raw/test_optim_database.db").resolve()
    config.output_dir = tmp_path
    return config


@pytest.fixture(
    params=[
        "painn.yaml",
        "schnet.yaml",
    ]
)
def spk_optim_config(request, tmp_path):
    cfg_dir_path = str((Path(nablaDFT.__path__[0]) / "../config").resolve())
    with initialize_config_dir(version_base="1.2", config_dir=cfg_dir_path, job_name=f"test_{request.param}"):
        config = compose(config_name="schnet_optim.yaml", overrides=[f"model={request.param}"])
    model_name = config.model.model_name
    config.pretrained = f"{model_name}_train_large"
    config.input_db = (Path(nablaDFT.__path__[0]) / "../tests/data/raw/test_optim_database.db").resolve()
    config.output_dir = tmp_path
    return config
