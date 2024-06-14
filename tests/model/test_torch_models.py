import pytest
from hydra.utils import instantiate

from .configs import model_configs


@pytest.mark.model
@pytest.mark.parametrize("model_cfg", model_configs())
def test_torch_model(model_cfg, dataset_pyg):
    # TODO: spill some asserts here on pyg_dataset
    torch_model_cfg = model_cfg['model']
    model = instantiate(torch_model_cfg)
    assert model
