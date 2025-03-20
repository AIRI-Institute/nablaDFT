import json
import logging
from importlib import import_module
from pathlib import Path
from typing import List, Mapping, Optional

import nablaDFT

DATASOURCES_LIST = sorted((Path(nablaDFT.__path__[0]) / "data/_metadata").glob("*.json"))
METADATA = [
    filename.name for filename in sorted((Path(nablaDFT.__path__[0]) / "data/_metadata/ds_meta").glob("*.json"))
]
logger = logging.getLogger()


class DatasourceRegistry:
    def __init__(self) -> None:
        pass

    def _get_url(self) -> str:
        pass

    def _download(self, name: str, path: Path) -> None:
        if path.exists():
            # TODO: add hashes comparison?
            logger.info(f"{name} already downloaded and saved in {path}")
            return


class DatasetRegistry:
    """Source of dataset splits links."""

    def get_dataset_url(self, task_type: str, name: str) -> Optional[str]:
        """Returns URL for given dataset split task and name.

        Args:
            task_type (str): type of task, must be one of :obj:["energy", "hamiltonian"].
            name (str): split name. Available splits can be listed with
                :meth:nablaDFT.registry.DatasetRegistry.list_datasets().
        """
        if task_type == "energy":
            url = self.energy_datasets.get(name, None)
        elif task_type == "hamiltonian":
            url = self.hamiltonian_datasets.get(name, None)
        else:
            raise NotImplementedError("Currently only two tasks supported: ['energy', 'hamiltonian']")
        if url is None:
            raise KeyError(f"Wrong name split: {name} or database file not found")
        return url

    def get_dataset_etag(self, task_type: str, name: str) -> Optional[str]:
        """Returns URL for given dataset split task and name.

        Args:
            task_type (str): type of task, must be one of :obj:["energy", "hamiltonian"].
            name (str): split name. Available splits can be listed with
                :meth:nablaDFT.registry.DatasetRegistry.list_datasets().
        """
        if task_type == "energy":
            etag = self.energy_datasets_etag.get(name, None)
        elif task_type == "hamiltonian":
            etag = self.hamiltonian_datasets_etag.get(name, None)
        else:
            etag = None
        return etag

    def list_datasets(self, task_type: str) -> Optional[List[str]]:
        """Returns available dataset splits for given task.

        Args:
            task_type (str): type of task, must be one of :obj:["energy", "hamiltonian"].
        """
        if task_type == "energy":
            return list(self.energy_datasets.keys())
        elif task_type == "hamiltonian":
            return list(self.hamiltonian_datasets.keys())
        else:
            return None


dataset_registry = DatasetRegistry()


def load_datasource():
    pass


def list_datasource():
    pass


def _build_datasource(type_cls: str, **kwargs):
    module = import_module("nablaDFT.data")
    cls_ = getattr(module, type_cls)
    datasource = cls_(**kwargs)
    return datasource
