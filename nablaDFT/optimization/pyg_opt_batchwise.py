import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np

from ase import io
from ase.db import connect
#from pyg_ase_interface import PYGAseInterface

from  pyg_ase_interface_batchwise import ASEBatchwiseLBFGS, BatchwiseCalculator
import utils

from hydra import compose, initialize
from omegaconf import OmegaConf

import tqdm

with initialize(version_base=None, config_path="../../config", job_name="test"):
    cfg = compose(config_name="dimenetplusplus_test.yaml")

ase_dir = 'ase_calcs'
if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

cutoff = 5.0
model_path = "/mnt/2tb/khrabrov/models/nablaDFT/checkpoints/DimeNet++/DimeNet++_dataset_train_100k_epoch=0258.ckpt"
db = connect("/mnt/2tb/data/nablaDFT/test_small_20k.db")
#atoms = connect("/mnt/2tb/data/nablaDFT/test10k_conformers_v2_formation_energy_w_forces.db").get(10).toatoms()
odb = connect("/mnt/2tb/data/nablaDFT/test_small_20k_dimenet_100k.db")
model = utils.load_model(cfg, model_path)
calculator = BatchwiseCalculator(model, device="cuda:1",  energy_unit="eV", position_unit="Ang")
batch_size = 128
db_len = len(db)
for batch_idx in tqdm.tqdm(range(0, db_len // batch_size)):
    atoms_list = [db.get(i + 1).toatoms() for i in range(batch_idx * batch_size, min(db_len, batch_size * (batch_idx + 1)))]
    optimizer = ASEBatchwiseLBFGS(calculator=calculator, master=True)
    optimizer.run(atoms_list, fmax=1e-4, steps=100)
    atoms_list = optimizer.atoms
    # print (atoms_list)
    for relative_id, i in enumerate(range(1 + batch_idx * batch_size, min(db_len, batch_size * (batch_idx + 1)))):
        row = db.get(i + 1)
        data = row.data
        data['model_energy'] = float(calculator.results['energy'][relative_id])
        odb.write(atoms_list[relative_id], data=data, moses_id=row.moses_id, conformation_id=row.conformation_id, smiles=row.smiles)
