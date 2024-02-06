import argparse
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

from  schnetpack_ase_interface_batchwise import ASEBatchwiseLBFGS, BatchwiseCalculator
from schnetpack.interfaces.ase_interface import AtomsConverter
import schnetpack.transform as trn

import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PYG batchwise optimization")
    parser.add_argument(
        "--database_path", type=str, help="database name"
    )

    parser.add_argument(
        "--outdatabase_path", type=str, help="output database name"
    )

    parser.add_argument(
        "--batch_size", type=int, help="batch size", default=128
    )

    parser.add_argument(
        "--model_ckpt_path", type=str, help="model ckpt path"
    )

    parser.add_argument(
        "--device", type=str, help="device", default="cuda:1"
    )
    args, unknown = parser.parse_known_args()

    ase_dir = 'ase_calcs'
    if not os.path.exists(ase_dir):
        os.mkdir(ase_dir)
    
    cutoff = 5.0
    #model_path = "/mnt/2tb/khrabrov/models/nablaDFT/checkpoints/DimeNet++/DimeNet++_dataset_train_100k_epoch=0258.ckpt"
    #db = connect("/mnt/2tb/data/nablaDFT/test_2k_random_optimized.db")
    db = connect(args.database_path)
    #atoms = connect("/mnt/2tb/data/nablaDFT/test10k_conformers_v2_formation_energy_w_forces.db").get(10).toatoms()
    odb = connect(args.outdatabase_path)
    model = torch.load(args.model_ckpt_path).to(args.device)
    atoms_converter = AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), device=args.device)
    calculator = BatchwiseCalculator(model, atoms_converter, device=args.device,  energy_unit="eV", position_unit="Ang")
    batch_size = args.batch_size
    
    db_len = len(db)
    for batch_idx in tqdm.tqdm(range(0, db_len // batch_size)):
        atoms_list = [db.get(i + 1).toatoms() for i in range(batch_idx * batch_size, min(db_len, batch_size * (batch_idx + 1)))]
        optimizer = ASEBatchwiseLBFGS(calculator=calculator, master=True, use_line_search=True)
        optimizer.run(atoms_list, fmax=1e-4, steps=100)
        atoms_list = optimizer.atoms
        # print (atoms_list)
        for relative_id, i in enumerate(range(batch_idx * batch_size, min(db_len, batch_size * (batch_idx + 1)))):
            row = db.get(i + 1)
            data = row.data
            data['model_energy'] = float(calculator.results['energy'][relative_id])
            odb.write(atoms_list[relative_id], data=data, moses_id=row.moses_id, conformation_id=row.conformation_id, smiles=row.smiles)
