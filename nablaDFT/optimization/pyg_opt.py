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
from pyg_ase_interface import PYGAseInterface

from hydra import compose, initialize
from omegaconf import OmegaConf
with initialize(version_base=None, config_path="../../config", job_name="test"):
    cfg = compose(config_name="dimenetplusplus_test.yaml")

ase_dir = 'ase_calcs'
if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

cutoff = 5.0
model_path = "/mnt/2tb/khrabrov/models/nablaDFT/checkpoints/DimeNet++/DimeNet++_dataset_train_100k_epoch=0258.ckpt"
atoms = connect("/mnt/2tb/data/nablaDFT/test10k_conformers_v2_formation_energy_w_forces.db").get(10).toatoms()
nablaDFT_ase = PYGAseInterface(
    ase_dir,
    ase_atoms=atoms,
    config=cfg,
    ckpt_path=model_path,
    energy_key="energy",
    force_key="forces",
    energy_unit="eV",
    position_unit="Ang",
    device="cuda:1",
    dtype=torch.float32,
)

nablaDFT_ase.optimize(fmax=1e-4, steps=100)
