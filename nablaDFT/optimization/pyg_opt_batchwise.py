import os
import argparse

from ase.db import connect

from  pyg_ase_interface_batchwise import ASEBatchwiseLBFGS, BatchwiseCalculator
import utils

from hydra import compose, initialize
from omegaconf import OmegaConf

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
        "--model_config_name", type=str, help="model config name", default="dimenetplusplus_test.yaml"
    )
    
    parser.add_argument(
        "--device", type=str, help="device", default="cuda:1"
    )
    args, unknown = parser.parse_known_args()

    with initialize(version_base=None, config_path="../../config", job_name="test"):
        cfg = compose(config_name=args.model_config_name)

    ase_dir = 'ase_calcs'
    if not os.path.exists(ase_dir):
        os.mkdir(ase_dir)
    
    cutoff = 5.0
    db = connect(args.database_path)
    odb = connect(args.outdatabase_path)
    model = utils.load_model(cfg, args.model_ckpt_path)
    calculator = BatchwiseCalculator(model, device=args.device,  energy_unit="eV", position_unit="Ang")
    batch_size = args.batch_size
    db_len = len(db)
    for batch_idx in tqdm.tqdm(range(0, db_len // batch_size)):
        atoms_list = [db.get(i + 1).toatoms() for i in range(batch_idx * batch_size, min(db_len, batch_size * (batch_idx + 1)))]
        optimizer = ASEBatchwiseLBFGS(calculator=calculator, master=True, use_line_search=True)
        optimizer.run(atoms_list, fmax=1e-4, steps=100)
        atoms_list = optimizer.atoms
        for relative_id, i in enumerate(range(batch_idx * batch_size, min(db_len, batch_size * (batch_idx + 1)))):
            row = db.get(i + 1)
            data = row.data
            data['model_energy'] = float(calculator.results['energy'][relative_id])
            odb.write(atoms_list[relative_id], data=data, moses_id=row.moses_id, conformation_id=row.conformation_id, smiles=row.smiles)
