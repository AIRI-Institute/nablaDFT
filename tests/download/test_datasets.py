import os
import pytest
from urllib import request as request

from nablaDFT.dataset import dataset_registry
from nablaDFT.utils import file_validation, get_file_size


@pytest.mark.download
@pytest.mark.parametrize("split", [
    "dataset_train_medium", "dataset_train_small", "dataset_train_tiny",
    "dataset_test_conformations_large", "dataset_test_conformations_medium",
    "dataset_test_conformations_small", "dataset_test_conformations_tiny",
    "dataset_test_trajectories_initial", "dataset_test_trajectories",
])
def test_energy_dataset_availability(tmp_path, split):
    save_path = os.path.join(tmp_path, split)
    url = dataset_registry.get_dataset_url("energy", split)
    request.urlretrieve(url, save_path)
    assert file_validation(save_path, dataset_registry.energy_datasets_etag[split])


@pytest.mark.download
@pytest.mark.parametrize("split", [
    "dataset_train_large", "dataset_train_full",
    "dataset_test_structures", "dataset_test_scaffolds",
    "dataset_test_conformations_full", "dataset_train_medium_trajectories",
    "dataset_trajectories_additional"
])
def test_energy_dataset_availability_large(split):
    # TODO: write me
    pass


@pytest.mark.download
@pytest.mark.parametrize("split", [
    "dataset_train_large", "dataset_train_medium",
    "dataset_train_small", "dataset_train_tiny",
    "dataset_test_structures", "dataset_test_scaffolds",
    "dataset_test_conformations_large", "dataset_test_conformations_medium",
    "dataset_test_conformations_small", "dataset_test_conformations_tiny"
])
def test_hamiltonian_dataset_availability(split):
    # TODO: write me
    pass
