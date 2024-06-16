import pytest
import torch
from torch_geometric.data import Batch

from tests.decorators import withCUDA

from nablaDFT import model_registry


@withCUDA
@pytest.mark.model
@pytest.mark.parametrize(
    "model_name",
    [
        "DimeNet++",
        "Equiformer-v2",
        "ESCN-OC",
        "GemNet-OC",
    ],
)
def test_pyg_model(model_name, dataset_pyg, device):
    model_name = model_name + "_train_tiny"
    model = model_registry.get_pretrained_model("torch", model_name).to(device)
    batch = Batch.from_data_list([dataset_pyg[0].to(device)])
    energy, forces = model(batch)
    assert energy.shape == batch.y.shape
    assert forces.shape == batch.forces.shape


@withCUDA
@pytest.mark.model
def test_graphormer(dataset_pyg, device):
    model_name = "Graphormer3D-small_train_tiny"
    model = model_registry.get_pretrained_model("torch", model_name).to(device)
    batch = Batch.from_data_list([dataset_pyg[0].to(device)])
    energy, forces, mask = model(batch)
    assert energy.shape == batch.y.shape
    assert forces.squeeze(dim=0).shape == batch.forces.shape
    assert torch.all(mask)


@pytest.mark.model
def test_hamiltonian_model(dataset_hamiltonian):
    model_name = "QHNet_train_tiny"
    model = model_registry.get_pretrained_model("torch", model_name)
    batch = Batch.from_data_list([dataset_hamiltonian[0]])
    # QHNet return block diagonal matrix
    output = model(batch)
    assert output.shape == batch.hamiltonian[0].shape
