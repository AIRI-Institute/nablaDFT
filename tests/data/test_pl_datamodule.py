import pytest
from nablaDFT.data import PLDataModule


@pytest.mark.data
def test_pl_datamodule_setup(pyg_dataset):
    """PyTorch Lightning basic test cases."""
    # fit stage
    datamodule = PLDataModule(pyg_dataset, train_size=0.8, batch_size=5, num_workers=2, shuffle=True)
    # check default kwargs
    assert datamodule.kwargs.get("pin_memory")
    assert datamodule.kwargs.get("shuffle")
    assert datamodule.kwargs.get("persistent_workers")
    datamodule.setup("fit")
    assert len(datamodule.dataset_train) == int(len(pyg_dataset) * 0.8)
    assert len(datamodule.dataset_val) == int(len(pyg_dataset) * 0.2)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    assert len(train_dataloader) == 4
    assert len(val_dataloader) == 1
    # test stage
    datamodule.setup("test")
    assert len(datamodule.dataset_test) == len(pyg_dataset)
    assert len(datamodule.test_dataloader()) == len(pyg_dataset) // 5
    # predict stage
    datamodule.setup("predict")
    assert len(datamodule.dataset_predict) == len(pyg_dataset)
    assert len(datamodule.predict_dataloader()) == len(pyg_dataset) // 5
