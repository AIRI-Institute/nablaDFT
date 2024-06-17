import json
import os
from urllib import request as request

import nablaDFT
import pytest
from nablaDFT import model_registry
from nablaDFT.utils.download import file_validation


def read_model_names():
    path = os.path.join(nablaDFT.__path__[0], "links/models_checkpoints.json")
    with open(path, "r") as fin:
        links = json.load(fin)
    urls = links["checkpoints"].keys()
    return [pytest.param(url) for url in urls]


@pytest.mark.download
@pytest.mark.parametrize("checkpoint_name", read_model_names())
def test_model_checkpoints(tmp_path, checkpoint_name):
    save_path = os.path.join(tmp_path, checkpoint_name)
    url = model_registry.get_pretrained_model_url(checkpoint_name)
    request.urlretrieve(url, save_path)
    assert file_validation(save_path, model_registry.get_pretrained_model_etag(checkpoint_name))
