import os

import pytest
import torch
from ase.db import connect
from nablaDFT.pipelines import run


@pytest.mark.pipeline
def test_train_pipeline(train_config, caplog):
    run(train_config)
    assert "`Trainer.fit` stopped: `max_epochs=3` reached." in caplog.messages[-1]


@pytest.mark.pipeline
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Can't test DDP with less than 2 devices")
def test_train_ddp_pipeline(train_ddp_config, caplog):
    run(train_ddp_config)
    assert "`Trainer.fit` stopped: `max_epochs=3` reached." in caplog.messages[-1]


@pytest.mark.pipeline
def test_test_pipeline(test_config, capsys, caplog):
    run(test_config)
    out, err = capsys.readouterr()
    lines = out.split("\n")
    for line in lines:
        if "test" in line:
            metric_name = [x.strip() for x in line.split("│")][1]
            if test_config.name in ["PaiNN", "SchNet"]:
                assert metric_name in ["test_energy_MAE", "test_forces_MAE", "test_loss"]
            else:
                assert metric_name in ["test/energy", "test/forces", "test/loss_epoch"]
                assert len(err) == 0


@pytest.mark.pipeline
def test_predict_pipeline(predict_config):
    run(predict_config)
    model_name = predict_config.model.model_name
    dataset_name = predict_config.dataset_name
    pred_db = connect(os.path.join(predict_config.output_dir, f"{model_name}_{dataset_name}.db"))
    assert len(pred_db) == 100
    for idx in range(1, len(pred_db) + 1):
        row = pred_db.get(idx)
        assert len(row.data["energy"]) == len(row.data["energy_pred"])
        assert row.data["forces"].shape == row.data["forces_pred"].shape
