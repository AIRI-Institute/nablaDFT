from typing import List

import json

import nablaDFT


class DatasetRegistry:
    """Source of dataset splits links."""
    def __init__(self):
        self.energy_datasets = {}
        self.hamiltonian_datasets = {}
        with open(nablaDFT.__path__[0] + "/links/energy_databases.json", "r") as fin:
            content = json.load(fin)
        for key in content.keys():
            for split_name in content[key].keys():
                self.energy_datasets[split_name] = content[key][split_name]
        with open(nablaDFT.__path__[0] + "/links/hamiltonian_databases.json", "r") as fin:
            content = json.load(fin)
        for key in content.keys():
            for split_name in content[key].keys():
                self.hamiltonian_datasets[split_name] = content[key][split_name]

    def get_dataset_url(self, task_type: str, name: str):
        """Returns URL for given dataset split task and name.

        Args:
            task_type (str): type of task, must be one of :obj:["energy", "hamiltonian"].
            name (str): split name. Available splits can be listed with :meth:nablaDFT.registry.DatasetRegistry.list_datasets
        """
        if task_type == "energy":
            url = self.energy_datasets.get(name, None)
        elif task_type == "hamiltonian":
            url = self.hamiltonian_datasets.get(name, None)
        else:
            raise NotImplementedError("Currently only two tasks supported: ['energy', 'hamiltonian']")
        if url is None:
            raise KeyError("Wrong name split or database file not found")
        return url

    def list_datasets(self, task_type: str) -> List[str]:
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
