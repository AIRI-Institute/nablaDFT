from typing import Optional, List

import numpy as np
import ase
import torch
from torch import nn
from schnetpack.units import convert_units
from schnetpack.interfaces.ase_interface import AtomsConverter, AtomsConverterError

from .opt_utils import atoms_list_to_PYG


class BatchwiseCalculator:
    """
    Base class calculator for neural network models for batchwise optimization.
    Args:
        model (nn.Module): loaded trained model.
        device (str): device used for calculations (default="cpu")
        energy_key (str): name of energies in model (default="energy")
        force_key (str): name of forces in model (default="forces")
        energy_unit (str): energy units used by model (default="eV")
        position_unit (str): position units used by model (default="Angstrom")
        dtype (torch.dtype): required data type for the model input (default: torch.float32)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):

        self.results = None
        self.atoms = None

        if type(device) == str:
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        self.energy_key = energy_key
        self.force_key = force_key

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy_key: self.energy_conversion,
            self.force_key: self.energy_conversion / self.position_conversion,
        }

        self.model = model
        self.model.to(device=self.device, dtype=self.dtype)

    def _requires_calculation(self, property_keys: List[str], atoms: List[ase.Atoms]):
        if self.results is None:
            return True
        for name in property_keys:
            if name not in self.results:
                return True
        if len(self.atoms) != len(atoms):
            return True
        for atom, atom_ref in zip(atoms, self.atoms):
            if atom != atom_ref:
                return True

    def get_forces(
        self, atoms: List[ase.Atoms], fixed_atoms_mask: Optional[List[int]] = None
    ) -> np.array:
        """Return atom's forces.
        Args:
            atoms (List[ase.Atoms]): list of ase.Atoms objects.
            fixed_atoms_mask (optional, List[int]): list of indices corresponding to atoms with positions fixed in space.
        """
        if self._requires_calculation(
            property_keys=[self.energy_key, self.force_key], atoms=atoms
        ):
            self.calculate(atoms)
        f = self.results[self.force_key]
        if fixed_atoms_mask is not None:
            f[fixed_atoms_mask] = 0.0
        return f

    def get_potential_energy(self, atoms: List[ase.Atoms]) -> float:
        if self._requires_calculation(property_keys=[self.energy_key], atoms=atoms):
            self.calculate(atoms)
        return self.results[self.energy_key]

    def calculate(self, atoms: List[ase.Atoms]) -> None:
        raise NotImplementedError


class PyGBatchwiseCalculator(BatchwiseCalculator):
    """Batchwise calculator for PyTorch Geometric models for batchwise optimization
    Args:
        model (nn.Module): loaded PyG model.
    """
       
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            model=model,
            device=device,
            energy_key=energy_key,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            dtype=dtype,
        )

    def calculate(self, atoms: List[ase.Atoms]) -> None:
        model_inputs = atoms_list_to_PYG(atoms, device=self.device)
        model_results = self.model(model_inputs)

        results = dict()
        results["energy"] = (
            model_results[0].cpu().data.numpy() * self.property_units["energy"]
        )
        results["forces"] = (
            model_results[1].cpu().data.numpy() * self.property_units["forces"]
        )

        self.results = results
        self.atoms = atoms.copy()


class SpkBatchwiseCalculator(BatchwiseCalculator):
    """Batchwise calculator for SchNetPack models for batchwise optimization.

    Args:
        model (nn.Module): loaded train schnetpack model.
        atoms_converter (AtomsConverter): Class used to convert ase Atoms objects to schnetpack input.
    """

    def __init__(
        self,
        model: nn.Module,
        atoms_converter: AtomsConverter,
        device: str = "cpu",
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            model=model,
            device=device,
            energy_key=energy_key,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            dtype=dtype,
        )
        self.atoms_converter = atoms_converter

    def calculate(self, atoms: List[ase.Atoms]) -> None:
        property_keys = list(self.property_units.keys())
        results = {}
        model_inputs = self.atoms_converter(atoms)
        model_results = self.model(model_inputs)

        results = {}
        # store model results in calculator
        for prop in property_keys:
            if prop in model_results:
                results[prop] = (
                    model_results[prop].detach().cpu().numpy()
                    * self.property_units[prop]
                )
            else:
                raise AtomsConverterError(
                    "'{:s}' is not a property of your model. Please "
                    "check the model "
                    "properties!".format(prop)
                )

        self.results = results
        self.atoms = atoms.copy()
