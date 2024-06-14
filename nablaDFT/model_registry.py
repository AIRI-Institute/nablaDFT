import os
from typing import List
import json
from urllib import request as request
import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import nablaDFT
from nablaDFT.utils import tqdm_download_hook, get_file_size


class ModelRegistry:
    """Source of pretrained model checkpoint links."""
    def __init__(self):
        with open(nablaDFT.__path__[0] + "/links/models_checkpoints.json", "r") as fin:
            content = json.load(fin)
        self.model_checkpoints = content['checkpoints']
        self.model_checkpoints_etag = content['etag']

    def get_pretrained_model_url(self, model_name: str) -> str:
        """Returns URL for given pretrained model name.

        Args:
            model_name (str): pretrained model name. Available models can be listed with :meth:nablaDFT.registry.ModelRegistry.list_models
        """
        url = self.model_checkpoints.get(model_name, None)
        if url:
            return self.model_checkpoints[model_name]
        else:
            raise KeyError(f"Wrong checkpoint name: {model_name}")

    def get_pretrained_model_etag(self, model_name: str) -> str:
        """Returns reference ETag for given pretrained model name.

        Args:
            model_name (str): pretrained model name. Available models can be listed with :meth:nablaDFT.registry.ModelRegistry.list_models
        """
        return self.model_checkpoints_etag[model_name]

    def list_models(self) -> List[str]:
        """Returns all available pretrained on nablaDFT model checkpoints."""
        return list(self.model_checkpoints.keys())


model_registry = ModelRegistry()


def download_pretrained_model(config: DictConfig) -> str:
    """Downloads pretrained model checkpoint from vault.

    Args:
        config (DictConfig): config for task. see r'config/' for examples.
    """
    model_name = config.get("name")
    ckpt_path = os.path.join(
        hydra.utils.get_original_cwd(),
        f"./checkpoints/{model_name}/{config.pretrained}.ckpt",
    )
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        os.makedirs(f"./checkpoints/{model_name}", exist_ok=True)
    url = model_registry.get_pretrained_model_url(config.pretrained)
    file_size = get_file_size(url)
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, total=file_size, desc=f"Downloading {model_name} checkpoint") as t:
        request.urlretrieve(url, ckpt_path, reporthook=tqdm_download_hook(t))
    logging.info(f"Downloaded {model_name} 100k checkpoint to {ckpt_path}")
    return ckpt_path
