from math import ceil

import pytest
import torch

from nablaDFT.dataset import ASENablaDFT
from .assertions import assert_shapes_spk


@pytest.mark.dataset
def test_spk_dataset_fit_dataloader(dataset_spk_params_train):
    dataset = ASENablaDFT(**dataset_spk_params_train)
    dataset.prepare_data()
    dataset.setup(dataset_spk_params_train["split"])
    dataloader_train, dataloader_val = (
        dataset.train_dataloader(),
        dataset.val_dataloader(),
    )
    train_batch = next(iter(dataloader_train))
    val_batch = next(iter(dataloader_train))
    assert len(dataloader_train) == ceil(
        len(dataset.train_dataset) / dataset.batch_size
    )
    assert len(dataloader_val) == ceil(len(dataset.val_dataset) / dataset.batch_size)
    for batch in [train_batch, val_batch]:
        assert_shapes_spk(batch)


@pytest.mark.dataset
def test_spk_dataset_test_dataloader(dataset_spk_params_test):
    dataset = ASENablaDFT(**dataset_spk_params_test)
    dataset.prepare_data()
    dataset.setup(dataset_spk_params_test["split"])
    dataloader_test = dataset.test_dataloader()
    assert len(dataloader_test) == ceil(len(dataset.test_dataset) / dataset.batch_size)
    batch = next(iter(dataloader_test))
    assert_shapes_spk(batch)


@pytest.mark.dataset
def test_spk_dataset_predict_dataloader(dataset_spk_params_predict):
    dataset = ASENablaDFT(**dataset_spk_params_predict)
    dataset.prepare_data()
    dataset.setup(dataset_spk_params_predict["split"])
    dataloader_predict = dataset.predict_dataloader()
    assert len(dataloader_predict) == ceil(
        len(dataset.predict_dataset) / dataset.batch_size
    )
    batch = next(iter(dataloader_predict))
    assert_shapes_spk(batch)
