import os
import pytest
from urllib import request as request

from nablaDFT.dataset import dataset_registry
from nablaDFT.utils import file_validation, get_file_size


@pytest.mark.download
@pytest.mark.parametrize(
    "split",
    [
        "dataset_train_medium",
        "dataset_train_small",
        "dataset_train_tiny",
        "dataset_test_conformations_large",
        "dataset_test_conformations_medium",
        "dataset_test_conformations_small",
        "dataset_test_conformations_tiny",
        "dataset_test_trajectories_initial",
        "dataset_test_trajectories",
    ],
)
def test_energy_dataset_availability(tmp_path, split):
    save_path = os.path.join(tmp_path, split)
    url = dataset_registry.get_dataset_url("energy", split)
    request.urlretrieve(url, save_path)
    assert file_validation(
        save_path, dataset_registry.get_dataset_etag("energy", split)
    )


# currently test only file size match
@pytest.mark.download
@pytest.mark.parametrize(
    "split, expected_file_size",
    [
        ("dataset_train_large", 2296700928),
        ("dataset_train_full", 20448043008),
        ("dataset_test_structures", 5274320896),
        ("dataset_test_scaffolds", 5172527104),
        ("dataset_test_conformations_full", 3732295680),
        ("dataset_train_medium_trajectories", 6523641856),
        ("dataset_trajectories_additional", 6297280512),
    ],
)
def test_energy_dataset_availability_large(split, expected_file_size):
    url = dataset_registry.get_dataset_url("energy", split)
    file_size = get_file_size(url)
    assert file_size == expected_file_size


# currently test only file size match
@pytest.mark.download
@pytest.mark.parametrize(
    "split, expected_file_size",
    [
        ("dataset_train_large", 709039628288),
        ("dataset_train_medium", 68388278272),
        ("dataset_train_small", 38103318528),
        ("dataset_train_tiny", 15118360576),
        ("dataset_test_structures", 1595255488512),
        ("dataset_test_scaffolds", 1579001356288),
        ("dataset_test_conformations_large", 123117957120),
        ("dataset_test_conformations_medium", 12038623232),
        ("dataset_test_conformations_small", 6854754304),
        ("dataset_test_conformations_tiny", 3099738112),
    ],
)
def test_hamiltonian_dataset_availability(split, expected_file_size):
    url = dataset_registry.get_dataset_url("hamiltonian", split)
    file_size = get_file_size(url)
    assert file_size == expected_file_size
