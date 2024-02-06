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
from schnetpack.data import ASEAtomsData
from schnetpack.data import AtomsDataModule
from schnetpack.interfaces import SpkCalculator, AseInterface

ase_dir = 'ase_calcs'
if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

cutoff = 5.0
model_path = "schnetpack_logs/schnet/logs/train_10k_trajectories_10k_6l_100_5_10_1e4_gauss_100000ep/best_inference_model"
molecule_path = "/home/khrabrov/mol.xyz"

nablaDFT_ase = spk.interfaces.AseInterface(
    molecule_path,
    ase_dir,
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=cutoff),
    energy_key="energy",
    force_key="forces",
    energy_unit="eV",
    position_unit="Ang",
    device="cuda:1",
    dtype=torch.float64,
)

nablaDFT_ase.optimize(fmax=1e-4, steps=100)
