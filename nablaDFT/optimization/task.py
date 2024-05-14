from typing import List

import tqdm
from ase.db import connect

from .optimizers import BatchwiseOptimizer


class BatchwiseOptimizeTask:
    """Use for batchwise molecules conformations geometry optimization.
    
    Args:
        input_datapath (str): path to ASE database with molecules.
        output_datapath (str): path to output database.
        optimizer (BatchwiseOptimizer): used for molecule geometry optimization.
        converter (AtomsConverter): optional, mandatory for SchNetPack models.
        batch_size (int): number of samples per batch.
    """

    def __init__(
        self,
        input_datapath: str,
        output_datapath: str,
        optimizer: BatchwiseOptimizer,
        batch_size: int,
    ) -> None:
        self.optimizer = optimizer
        self.bs = batch_size
        self.data_db_conn = None
        self.out_db_conn = None
        self._open(input_datapath, output_datapath)

    def optimize_batch(self, atoms_list: List):
        self.optimizer.initialize()
        self.optimizer.run(atoms_list, fmax=1e-4, steps=100)
        atoms_list = self.optimizer.atoms
        return atoms_list

    def run(self):
        db_len = len(self.data_db_conn)
        batch_count = db_len // self.bs
        if db_len % self.bs:
            batch_count += 1
        for batch_idx in tqdm.tqdm(range(batch_count)):
            atoms_list = [
                self.data_db_conn.get(i + 1).toatoms()
                for i in range(
                    batch_idx * self.bs, min(db_len, self.bs * (batch_idx + 1))
                )
            ]
            atoms_list = self.optimize_batch(atoms_list)
            for relative_id, i in enumerate(
                range(batch_idx * self.bs, min(db_len, self.bs * (batch_idx + 1)))
            ):
                row = self.data_db_conn.get(i + 1)
                data = row.data
                data["model_energy"] = float(
                    self.optimizer.calculator.results["energy"][relative_id]
                )
                self.out_db_conn.write(
                    atoms_list[relative_id],
                    data=data,
                    moses_id=row.moses_id,
                    conformation_id=row.conformation_id,
                    smiles=row.smiles,
                )

    def _open(self, input_datapath, output_datapath):
        self.data_db_conn = connect(input_datapath)
        self.out_db_conn = connect(output_datapath)
