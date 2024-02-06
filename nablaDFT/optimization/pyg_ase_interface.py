import os

import ase
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations

import torch
import schnetpack
import logging
from copy import deepcopy

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.batch import Batch

from schnetpack.units import convert_units

from typing import Optional, List, Union, Dict
from ase import Atoms
from omegaconf import DictConfig

import utils

log = logging.getLogger(__name__)


class PYGCalculatorError(Exception):
    pass

def atoms_to_PYG(ase_atoms, device):
    z = torch.from_numpy(ase_atoms.numbers).long()
    positions = torch.from_numpy(ase_atoms.positions).float()
    batch = Batch.from_data_list([Data(z=z, pos=positions)]).to(device)
    return batch.pos, batch.z, batch.batch

            
class PYGCalculator(Calculator):
    """
    ASE calculator for pytorch geometric machine learning models.

    """

    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(
        self,
        config: DictConfig,
        ckpt_path: str,
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: Union[str, float] = "Hartree",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):

        """
        Args:
            model_file (str): path to trained model
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            energy_unit (str, float): energy units used by model (default="kcal/mol")
            position_unit (str, float): position units used by model (default="Angstrom")
            device (torch.device): device used for calculations (default="cpu")
            dtype (torch.dtype): select model precision (default=float32)
            converter (callable): converter used to set up input batches
            additional_inputs (dict): additional inputs required for some transforms in the converter.
            **kwargs: Additional arguments for basic ase calculator class
        """
        Calculator.__init__(self, **kwargs)

        self.energy_key = energy_key
        self.force_key = force_key
        self.device = device
        # Mapping between ASE names and model outputs
        self.property_map = {
            self.energy: energy_key,
            self.forces: force_key,
        }

        self.model = utils.load_model(config, ckpt_path)
        self.model.to(device=device, dtype=dtype)

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy: self.energy_conversion,
            self.forces: self.energy_conversion / self.position_conversion,
        }

        # Container for basic ml model ouputs
        self.model_results = None

    def _load_model(self, model_file: str):
        """

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        """

        log.info("Loading model from {:s}".format(model_file))
        # load model and keep it on CPU, device can be changed afterwards
        model = utils.load_model(config, ckpt_path)
        model = model.eval()

        return model

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): select properties computed and stored to results.
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)

            model_inputs = atoms_to_PYG(atoms, device=self.device)
            model_results = self.model(*model_inputs)
            
            results = dict()
            results["energy"] = ( model_results[0].cpu().data.numpy().item() * self.property_units["energy"] )
            results["forces"] = ( model_results[1].cpu().data.numpy() * self.property_units["forces"] )
            self.results = results
            self.model_results = model_results


class PYGAseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    """

    def __init__(
        self,
        working_dir: str,
        config: DictConfig,
        ckpt_path: str,
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: Union[str, float] = "Hartree",
        position_unit: Union[str, float] = "Angstrom",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        optimizer_class: type = QuasiNewton,
        fixed_atoms: Optional[List[int]] = None,
        ase_atoms: Optional = None,
        ):
        """
        Args:
            molecule_path: Path to initial geometry
            working_dir: Path to directory where files should be stored
            model_file (str): path to trained model
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            energy_unit (str, float): energy units used by model (default="kcal/mol")
            position_unit (str, float): position units used by model (default="Angstrom")
            device (torch.device): device used for calculations (default="cpu")
            dtype (torch.dtype): select model precision (default=float32)
            optimizer_class (ase.optimize.optimizer): ASE optimizer used for structure relaxation.
            fixed_atoms (list(int)): list of indices corresponding to atoms with positions fixed in space.

        """
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = ase_atoms

        # Apply position constraints
        if fixed_atoms:
            c = FixAtoms(fixed_atoms)
            self.molecule.set_constraint(constraint=c)

        # Set up optimizer
        self.optimizer_class = optimizer_class

        # Set up calculator
        calculator = PYGCalculator(
            config=config,
            ckpt_path=ckpt_path,
            energy_key=energy_key,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            device=device,
            dtype=dtype,
        )

        self.molecule.set_calculator(calculator)

        self.dynamics = None

    def save_molecule(self, name: str, file_format: str = "xyz", append: bool = False):
        """
        Save the current molecular geometry.

        Args:
            name: Name of save-file.
            file_format: Format to store geometry (default xyz).
            append: If set to true, geometry is added to end of file (default False).
        """
        molecule_path = os.path.join(
            self.working_dir, "{:s}.{:s}".format(name, file_format)
        )
        write(molecule_path, self.molecule, format=file_format, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule("single_point", file_format="xyz")

    def init_md(
        self,
        name: str,
        time_step: float = 0.5,
        temp_init: float = 300,
        temp_bath: Optional[float] = None,
        reset: bool = False,
        interval: int = 1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.

        Args:
            name: Basic name of logfile and trajectory
            time_step: Time step in fs (default=0.5)
            temp_init: Initial temperature of the system in K (default is 300)
            temp_bath: Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset: Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval: Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if self.dynamics is None or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(
                self.molecule,
                time_step * units.fs,
                temp_bath * units.kB,
                1.0 / (100.0 * units.fs),
            )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "{:s}.log".format(name))
        trajfile = os.path.join(self.working_dir, "{:s}.traj".format(name))
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(
        self,
        temp_init: float = 300,
        remove_translation: bool = True,
        remove_rotation: bool = True,
    ):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init: Initial temperature in Kelvin (default 300)
            remove_translation: Remove translation components of velocity (default True)
            remove_rotation: Remove rotation components of velocity (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps: int):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.

        Args:
            steps: Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError(
                "Dynamics need to be initialized using the" " 'setup_md' function"
            )

        self.dynamics.run(steps)

    def optimize(self, fmax: float = 1.0e-2, steps: int = 1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
            fmax: Maximum residual force change (default 1.e-2)
            steps: Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = self.optimizer_class(
            self.molecule,
            trajectory="{:s}.traj".format(optimize_file),
            restart=None,
        )
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name, file_format="extxyz")

    def compute_normal_modes(self, write_jmol: bool = True):
        """
        Use ase calculator to compute numerical frequencies for the molecule

        Args:
            write_jmol: Write frequencies to input file for visualization in jmol (default=True)
        """
        freq_file = os.path.join(self.working_dir, "normal_modes")

        # Compute frequencies
        frequencies = Vibrations(self.molecule, name=freq_file)
        frequencies.run()

        # Print a summary
        frequencies.summary()

        # Write jmol file if requested
        if write_jmol:
            frequencies.write_jmol()

