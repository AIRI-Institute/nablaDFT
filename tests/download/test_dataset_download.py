from pathlib import Path

import pytest
from nablaDFT.data import ASENablaDFT, PyGNablaDFT


@pytest.mark.download
def test_pyg_download_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dirpath = "./datasets/nablaDFT/train"
    PyGNablaDFT(dirpath, "dataset_train_tiny", "train")
    assert (Path(tmp_path) / dirpath / "raw/dataset_train_tiny.db").exists()
    assert (Path(tmp_path) / dirpath / "processed/dataset_train_tiny_train.pt").exists()
    monkeypatch.undo()


@pytest.mark.download
def test_spk_downloading_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dirpath = "./datasets/nablaDFT/train"
    dataset = ASENablaDFT("dataset_train_tiny", "train", dirpath, batch_size=2)
    dataset.prepare_data()
    assert (Path(tmp_path) / dirpath / "dataset_train_tiny.db").exists()
    monkeypatch.undo()


@pytest.mark.download
def test_pyg_download_local_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dirpath = "."
    PyGNablaDFT(dirpath, "dataset_train_tiny", "train")
    assert (Path(tmp_path) / "raw/dataset_train_tiny.db").exists()
    assert (Path(tmp_path) / "processed/dataset_train_tiny_train.pt").exists()
    monkeypatch.undo()


@pytest.mark.download
def test_spk_download_local_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dirpath = "."
    dataset = ASENablaDFT("dataset_train_tiny", "train", dirpath, batch_size=2)
    dataset.prepare_data()
    assert (Path(tmp_path) / "dataset_train_tiny.db").exists()
    monkeypatch.undo()
