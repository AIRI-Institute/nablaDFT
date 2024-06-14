import pytest
import torch
from torch_geometric.data import Batch
from hydra.utils import instantiate

from tests.configs import read_model_cfg
from tests.decorators import withCUDA


@withCUDA
@pytest.mark.model
@pytest.mark.parametrize(
    "model_cfg_name",
    [
        "dimenetplusplus",
        "equiformer_v2_oc20",
        "escn-oc",
        "gemnet-oc",
        "painn-oc",
    ],
)
def test_pyg_model(model_cfg_name, dataset_pyg, device):
    model_cfg = read_model_cfg(model_cfg_name)
    torch_model_cfg = model_cfg["model"]
    model = instantiate(torch_model_cfg).to(device)
    batch = Batch.from_data_list([dataset_pyg[0].to(device)])
    energy, forces = model(batch)
    assert energy.shape == batch.y.shape
    assert forces.shape == batch.forces.shape


@withCUDA
@pytest.mark.model
def test_graphormer(dataset_pyg, device):
    model_cfg = read_model_cfg("graphormer3d-small")
    torch_model_cfg = model_cfg["model"]
    model = instantiate(torch_model_cfg).to(device)
    batch = Batch.from_data_list([dataset_pyg[0].to(device)])
    energy, forces, mask = model(batch)
    assert energy.shape == batch.y.shape
    assert forces.squeeze(dim=0).shape == batch.forces.shape
    assert torch.all(mask)


@pytest.mark.model
@pytest.mark.parametrize("model_cfg_name", ["qhnet"])
def test_hamiltonian_model(model_cfg_name, dataset_hamiltonian):
    model_cfg = read_model_cfg(model_cfg_name)
    torch_model_cfg = model_cfg["model"]
    model = instantiate(torch_model_cfg)
    batch = Batch.from_data_list([dataset_hamiltonian[0]])
    # QHNet return block diagonal matrix
    output = model(batch)
    assert output.shape == batch.hamiltonian[0].shape
