from pathlib import Path

import pytest
from ase.db import connect
from nablaDFT.pipelines import run


@pytest.mark.optimization
def test_pyg_optimization(pyg_optim_config):
    run(pyg_optim_config)
    output_datapath = Path(pyg_optim_config.output_dir, f"{pyg_optim_config.name}_{pyg_optim_config.dataset_name}.db")
    db = connect(output_datapath)
    for idx in range(1, len(db) + 1):
        row = db.get(idx)
        assert isinstance(row.data["model_energy"], list)
        assert row.data["model_energy"][0] < 0
        assert row.data["forces"].shape == row.data["model_forces"].shape


@pytest.mark.optimization
def test_spk_optimization(spk_optim_config):
    run(spk_optim_config)
    output_datapath = Path(spk_optim_config.output_dir, f"{spk_optim_config.name}_{spk_optim_config.dataset_name}.db")
    db = connect(output_datapath)
    for idx in range(1, len(db) + 1):
        row = db.get(idx)
        assert row.data["energy"][0] > row.data["model_energy"][0]
        assert row.data["forces"].shape == row.data["model_forces"].shape
