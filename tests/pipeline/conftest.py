from pathlib import Path

import nablaDFT
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


@pytest.fixture(
    params=[
        "dimenetplusplus.yaml",
        "equiformer_v2_oc20.yaml",
        "escn-oc.yaml",
        "gemnet-oc.yaml",
        "graphormer3d.yaml",
        "painn.yaml",
        "schnet.yaml",
    ]
)
def train_config(request):
    cfg_dir_path = str((Path(nablaDFT.__path__[0]) / "../config").resolve())
    with initialize_config_dir(version_base="1.2", config_dir=cfg_dir_path, job_name=f"test_{request.param}"):
        config = compose(config_name=request.param)
    config.loggers = {}
    config.callbacks = {}
    with open_dict(config):
        config.trainer.max_epochs = 3
        config.trainer.enable_checkpointing = False
        del config.trainer.max_steps
    config.root = (Path(nablaDFT.__path__[0]) / "../tests/data").resolve()
    config.dataset_name = "test_database"
    return config


@pytest.fixture(
    params=[
        "dimenetplusplus.yaml",
        "equiformer_v2_oc20.yaml",
        "escn-oc.yaml",
        "gemnet-oc.yaml",
        "graphormer3d.yaml",
        "painn.yaml",
        "schnet.yaml",
    ]
)
def test_config(request):
    cfg_dir_path = (Path(nablaDFT.__path__[0]) / "../config").resolve()
    cfg = OmegaConf.load(cfg_dir_path / request.param)
    datamodule_cfg_name = cfg.defaults[1]["datamodule"].split(".")[0]
    with initialize_config_dir(version_base="1.2", config_dir=str(cfg_dir_path), job_name=f"test_{request.param}"):
        config = compose(
            config_name=request.param, overrides=[f"datamodule={datamodule_cfg_name}_test.yaml", "trainer=test.yaml"]
        )
    config.loggers = {}
    config.callbacks = {}
    config.job_type = "test"
    config.root = (Path(nablaDFT.__path__[0]) / "../tests/data").resolve()
    config.dataset_name = "test_database"
    return config


@pytest.fixture(
    params=[
        "gemnet-oc.yaml",
        "dimenetplusplus.yaml",
        "equiformer_v2_oc20.yaml",
        "escn-oc.yaml",
        "graphormer3d.yaml",
        "painn.yaml",
        "schnet.yaml",
    ]
)
def predict_config(request, tmp_path):
    cfg_dir_path = (Path(nablaDFT.__path__[0]) / "../config").resolve()
    cfg = OmegaConf.load(cfg_dir_path / request.param)
    datamodule_cfg_name = cfg.defaults[1]["datamodule"].split(".")[0]
    with initialize_config_dir(version_base="1.2", config_dir=str(cfg_dir_path), job_name=f"test_{request.param}"):
        config = compose(
            config_name=request.param, overrides=[f"datamodule={datamodule_cfg_name}_test.yaml", "trainer=test.yaml"]
        )
    config.loggers = {}
    config.callbacks = {}
    config.job_type = "predict"
    with open_dict(config):
        config.output_dir = tmp_path / "predictions_test"
    config.root = (Path(nablaDFT.__path__[0]) / "../tests/data").resolve()
    config.dataset_name = "test_database"
    return config
