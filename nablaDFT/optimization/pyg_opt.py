import argparse
import os

import hydra.utils
import torch
from hydra import compose, initialize
from pyg_ase_interface import PYGAseInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyG molecule optimization")
    parser.add_argument("--molecule_path", type=str, help="database name")
    parser.add_argument("--model_ckpt_path", type=str, help="model ckpt path")

    parser.add_argument(
        "--model_config_name",
        type=str,
        help="model config name",
    )
    parser.add_argument("--device", type=str, help="device", default="cuda:0")
    args, unknown = parser.parse_known_args()
    with initialize(version_base=None, config_path="../../config", job_name="test"):
        cfg = compose(config_name=args.model_config_name)

    ase_dir = "ase_calcs"
    if not os.path.exists(ase_dir):
        os.mkdir(ase_dir)

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cutoff = 5.0
    nablaDFT_ase = PYGAseInterface(
        molecule_path=args.molecule_path,
        model=model,
        working_dir=ase_dir,
        config=cfg,
        ckpt_path=args.model_ckpt_path,
        energy_key="energy",
        force_key="forces",
        energy_unit="eV",
        position_unit="Ang",
        device=args.device,
        dtype=torch.float32,
    )
    nablaDFT_ase.optimize(fmax=1e-4, steps=100)
