import logging
import argparse
from pathlib import Path

from tqdm import tqdm
from ase.db import connect
from xtb.libxtb import VERBOSITY_MUTED
from xtb.interface import Calculator, Param


ATOMNUM_TO_ELEM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}
ATOM_ENERGIES_XTB = {
    "H": -0.393482763936,
    "C": -1.793296371365,
    "O": -3.767606950376,
    "N": -2.605824161279,
    "F": -4.619339964238,
    "S": -3.146456870402,
    "P": -2.374178794732,
    "Cl": -4.482525134961,
    "Br": -4.048339371234,
    "I": -3.779630263390,
}
CONV_FACTOR = 0.52917720859


logger = logging.getLogger(__name__)


def atomic_energy(atoms):
    atom_symbol = [ATOMNUM_TO_ELEM[atom_num] for atom_num in atoms]
    atomic_energy = [ATOM_ENERGIES_XTB[atom] for atom in atom_symbol]
    return sum(atomic_energy)

def calculate_gfn2(atoms, positions):
    calc = Calculator(Param.GFN2xTB, atoms, positions / CONV_FACTOR)
    calc.set_accuracy(0.0001)
    calc.set_max_iterations(100)
    calc.set_verbosity(VERBOSITY_MUTED)
    res = calc.singlepoint()
    energy = res.get_energy() - atomic_energy(atoms) 
    forces = res.get_gradient() * CONV_FACTOR
    return energy, forces

def generate_gfn2xtb_db(input_db_path, output_db_path):
    db = connect(input_db_path)
    odb = connect(output_db_path)
    for row in tqdm(db.select(), desc="Generate GFN2-xTB database", total=len(db)):
        gfn2_energy, gfn2_force = calculate_gfn2(row.numbers, row.positions)
        data = {"energy": gfn2_energy, "forces": gfn2_force}
        odb.write(row, data=data)


def generate_delta_db(
        dft_db_path: str,
        gfn_db_path: str,
        output_db_path: str
    ):
    dft_db = connect(dft_db_path)
    gfn_db = connect(gfn_db_path)
    with connect(output_db_path) as odb:
        for idx in tqdm(range(1, len(dft_db) + 1), desc="Generate Delta database", total=(len(dft_db)+1)):
            row = dft_db.get(idx) # used for new db
            dft_data = row.data
            gfn_data = gfn_db.get(idx).data
            data = {
                "energy": dft_data["energy"] - gfn_data["energy"],
                "forces": dft_data["forces"] - gfn_data["forces"]
            }
            odb.write(row, data=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_db", type=str, help="path to ASE database with atoms and positions"
    )
    args, unknown = parser.parse_known_args()
    suffix = Path(args.input_db).suffix
    input_path_wo_suffix = args.input_db[:-len(suffix)]
    gfn2xtb_db_path = input_path_wo_suffix + "_gfn2xtb.db"
    delta_db_path = input_path_wo_suffix + "_delta.db"
    generate_gfn2xtb_db(args.input_db, gfn2xtb_db_path)
    logger.info(f"Generate GFN2-xTB database at {gfn2xtb_db_path}")
    generate_delta_db(args.input_db, gfn2xtb_db_path, delta_db_path)
    logger.info(f"Generate Delta database at {delta_db_path}")
