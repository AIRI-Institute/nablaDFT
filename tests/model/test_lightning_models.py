import pytest
from nablaDFT import model_registry
from torch_geometric.data import Batch

from tests.decorators import withCUDA


@withCUDA
@pytest.mark.model
@pytest.mark.parametrize(
    "model_name",
    ["DimeNet++", "Equiformer-v2", "ESCN-OC", "GemNet-OC", "Graphormer3D-small"],
)
def test_pyg_lightning_model(model_name, dataset_pyg, device):
    model_name = model_name + "_train_tiny"
    model = model_registry.get_pretrained_model("lightning", model_name).to(device)
    batch = Batch.from_data_list([dataset_pyg[0].to(device)])
    energy, forces = model(batch)
    assert energy.shape == batch.y.shape
    assert forces.shape == batch.forces.shape


@pytest.mark.model
def test_lightning_qhnet(dataset_hamiltonian):
    model_name = "QHNet_train_tiny"
    model = model_registry.get_pretrained_model("lightning", model_name).to("cpu")
    batch = Batch.from_data_list([dataset_hamiltonian[0].to("cpu")])
    # QHNet return block diagonal matrix
    output = model(batch)
    assert output.shape == batch.hamiltonian[0].shape
