from typing import List, Optional, Dict
import torch
import os
from urllib import request as request
from ase.db import connect

import schnetpack.properties as structure
from schnetpack.data import *
import json
import sys
sys.path.append('../')
from phisnet.training.hamiltonian_dataset import HamiltonianDataset
from phisnet.training.sqlite_database import HamiltonianDatabase

class ASENablaDFT(AtomsDataModule):

    def __init__(self, dataset_name: str, format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE, **kwargs):

        super().__init__(format=format, **kwargs)
        self.dataset_name = dataset_name
        
    def prepare_data(self):
        datapath_with_no_suffix = os.path.splitext(self.datapath)[0]
        suffix = os.path.splitext(self.datapath)[1]
        if not os.path.exists(datapath_with_no_suffix):
            os.makedirs(datapath_with_no_suffix)
        f = open('../links/energy_databases.json')
        data = json.load(f)
        url = data['train_databases'][self.dataset_name]
        f.close()
        self.datapath = datapath_with_no_suffix + "/" + self.dataset_name + suffix
        request.urlretrieve(url, self.datapath)    
        with connect(self.datapath) as ase_db:
            if not ase_db.metadata:
                ase_db.metadata = {"_distance_unit":"Bohr", "_property_unit_dict": {"energy": "Hartree"}}
            dataset_length = len(ase_db)
            self.num_train = int(dataset_length * 0.9)
            self.num_val = int(dataset_length * 0.1)
        self.dataset = load_dataset(self.datapath, self.format)
            
class HamiltonianNablaDFT(HamiltonianDataset):
    
    def __init__(self, datapath,dataset_name, max_batch_orbitals=1200, max_batch_atoms=150, max_squares=4802, subset=None, dtype=torch.float32):

        self.dtype = dtype
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        f = open('../links/hamiltonian_databases.json')
        data = json.load(f)
        url = data['train_databases'][dataset_name]
        f.close()
        filepath = datapath + "/" + dataset_name + ".db"
        request.urlretrieve(url, filepath) 
        self.database = HamiltonianDatabase(filepath)
        max_orbitals =[]
        for z in self.database.Z:
            max_orbitals.append(tuple((int(z),int(l)) for l in self.database.get_orbitals(z)))
        max_orbitals = tuple(max_orbitals)
        self.max_orbitals = max_orbitals
        self.max_batch_orbitals = max_batch_orbitals
        self.max_batch_atoms = max_batch_atoms
        self.max_squares = max_squares
        self.subset = None
        if subset:
            self.subset = np.load(subset)

def nablaDFT(type_of_nn: str, *args, **kwargs):
    valid = {"ASE","Hamiltonian"}
    if type_of_nn not in valid:
        raise ValueError("results: type of nn must be one of %r." % valid)
    if type_of_nn == "ASE":
        return ASENablaDFT(*args, **kwargs)
    else:
        return HamiltonianNablaDFT(*args, **kwargs)