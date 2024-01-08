from copy import deepcopy
import os
import pickle
import time

import ase
import numpy as np
from math import sqrt
from os.path import isfile

from ase.optimize.optimize import Dynamics
from ase.parallel import world, barrier
from ase.io import write
from ase import Atoms
# from ase.utils.linesearch import LineSearch

from typing import Dict, Optional, List, Tuple, Union

import torch
from torch import nn
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.batch import Batch
import schnetpack
from schnetpack.units import convert_units
from schnetpack.interfaces.ase_interface import AtomsConverter


import numpy as np
pymin = min
pymax = max


def np_scatter_add(updates, indices, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    #updates = updates.ravel()
    # print(updates.shape, indices.shape, shape)
    np.add.at(target, indices, updates)
    return target


class LineSearch:
    def __init__(self,  xtol=1e-14):

        self.xtol = xtol
        self.task = 'START'
        self.fc = 0
        self.gc = 0
        self.case = 0
        self.old_stp = 0

    def _line_search(self, func, myfprime, xk, pk, gfk, old_fval, old_old_fval,
                     n_configs, ats_mask, batch, n_ats_per_config, n_ats,
                     maxstep=.2, c1=.23, c2=0.46, xtrapl=1.1, xtrapu=4.,
                     stpmax=50., stpmin=1e-8, max_abs_step=100, args=()):
        self.stpmin = stpmin
        self.pk = pk
        # ??? p_size = np.sqrt((pk **2).sum())
        self.stpmax = stpmax
        self.xtrapl = xtrapl
        self.xtrapu = xtrapu
        self.maxstep = maxstep
        self.phi0 = old_fval
        
        #alpha1 = pymin(maxstep,1.01*2*(phi0-old_old_fval)/derphi0)
        alpha1 = np.ones(n_configs)
        self.no_update = False

        if isinstance(myfprime,type(())):
            # eps = myfprime[1]
            fprime = myfprime[0]
            # ??? newargs = (f,eps) + args
            gradient = False
        else:
            fprime = myfprime
            newargs = args
            gradient = True

        fval = old_fval
        gval = gfk
        
        steps = np.ones(n_configs)
        old_steps = np.zeros(n_configs)
        self.tasks = ['START'] * n_configs

        self.isave = np.zeros((n_configs, 2), np.intc)
        self.dsave = np.zeros((n_configs, 13), float)
        for at_idx in range(n_configs):
            p_at = self.pk[ats_mask[at_idx]].ravel()
            p_size = np.sqrt((p_at ** 2).sum())
            if p_size <= np.sqrt(n_ats_per_config[at_idx] * 1e-10):
                p_at /=  (p_size / np.sqrt(n_ats * 1e-10))
            self.pk[ats_mask[at_idx]] = p_at.reshape(-1, 3)
        abs_step = 0
        while True:
            abs_step += 1
            if abs_step > max_abs_step:
                break
            for at_idx in range(n_configs):
                phi0 = self.phi0[at_idx]
                pk = self.pk[ats_mask[at_idx]].ravel()
                derphi0 = np.dot(gval[ats_mask[at_idx]].ravel(), pk)
                self.dim = len(pk)
                self.gms = np.sqrt(self.dim) * maxstep
                stp = self.step(steps[at_idx], phi0, derphi0, c1, c2,
                                pk, old_steps[at_idx], at_idx,
                                self.xtol, self.isave, self.dsave)
                # print (stp, self.case, steps[at_idx], phi0, derphi0, c1, c2,
                #                old_steps[at_idx], 
                #                self.xtol, self.isave, self.dsave)
                #print (stp, self.case)
                old_steps[at_idx] = steps[at_idx]
                if self.tasks[at_idx] in ['FG', 'CONVERGENCE'] and not self.no_update:
                    steps[at_idx] = stp
                else:
                    steps[at_idx] = self.determine_step_(pk)
            # print(self.tasks)        
            if not np.any([x[:2] == 'FG' for x in self.tasks]):
                break
            alpha1 = np.repeat(steps, n_ats_per_config).reshape(self.pk.shape[0], -1)
            fval = func(xk + self.pk * alpha1, *args)
            self.fc += 1
            gval = fprime(xk + self.pk * alpha1, *newargs)
            if gradient: self.gc += 1
            else: self.fc += len(xk) + 1
            self.phi0 = fval
            self.old_stp = alpha1

        alpha1 = np.repeat(steps, n_ats_per_config)
        return alpha1, fval, old_fval, self.no_update

    def step(self, stp, f, g, c1, c2, pk, old_stp, at_idx, xtol, isave, dsave):
        if self.tasks[at_idx][:5] == 'START':
            # Check the input arguments for errors.
            if stp < self.stpmin:
                self.tasks[at_idx] = 'ERROR: STP .LT. minstep'
            if stp > self.stpmax:
                self.tasks[at_idx] = 'ERROR: STP .GT. maxstep'
            if g >= 0:
                self.tasks[at_idx] = 'ERROR: INITIAL G >= 0'
            if c1 < 0:
                self.tasks[at_idx] = 'ERROR: c1 .LT. 0'
            if c2 < 0:
                self.tasks[at_idx] = 'ERROR: c2 .LT. 0'
            if xtol < 0:
                self.tasks[at_idx] = 'ERROR: XTOL .LT. 0'
            if self.stpmin < 0:
                self.tasks[at_idx] = 'ERROR: minstep .LT. 0'
            if self.stpmax < self.stpmin:
                self.tasks[at_idx] = 'ERROR: maxstep .LT. minstep'
            if self.tasks[at_idx][:5] == 'ERROR':
                return stp

            # Initialize local variables.
            self.bracket = False
            stage = 1
            finit = f
            ginit = g
            gtest = c1 * ginit
            width = self.stpmax - self.stpmin
            width1 = width / .5
#           The variables stx, fx, gx contain the values of the step,
#           function, and derivative at the best step.
#           The variables sty, fy, gy contain the values of the step,
#           function, and derivative at sty.
#           The variables stp, f, g contain the values of the step,
#           function, and derivative at stp.
            stx = 0
            fx = finit
            gx = ginit
            sty = 0
            fy = finit
            gy = ginit
            stmin = 0
            stmax = stp + self.xtrapu * stp
            self.tasks[at_idx] = 'FG'
            self.save(at_idx, (stage, ginit, gtest, gx,
                       gy, finit, fx, fy, stx, sty,
                       stmin, stmax, width, width1))
            stp = self.determine_step(stp, old_stp, pk)
            #return stp, f, g
            return stp
        else:
            if self.isave[at_idx][0] == 1:
                self.bracket = True
            else:
                self.bracket = False
            stage = self.isave[at_idx][1]
            (ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, \
             width, width1) =self.dsave[at_idx]

#           If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
#           algorithm enters the second stage.
            ftest = finit + stp * gtest
            if stage == 1 and f < ftest and g >= 0.:
                stage = 2

#           Test for warnings.
            if self.bracket and (stp <= stmin or stp >= stmax):
                self.tasks[at_idx] = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
            if self.bracket and stmax - stmin <= self.xtol * stmax:
                self.tasks[at_idx] = 'WARNING: XTOL TEST SATISFIED'
            if stp == self.stpmax and f <= ftest and g <= gtest:
                self.tasks[at_idx] = 'WARNING: STP = maxstep'
            if stp == self.stpmin and (f > ftest or g >= gtest):
                self.tasks[at_idx] = 'WARNING: STP = minstep'

#           Test for convergence.
            # print (f, ftest, abs(g), c2 * (- ginit))
            if f <= ftest and abs(g) <= c2 * (- ginit):
                self.tasks[at_idx] = 'CONVERGENCE'

#           Test for termination.
            if self.tasks[at_idx][:4] == 'WARN' or self.tasks[at_idx][:4] == 'CONV':
                self.save(at_idx, (stage, ginit, gtest, gx,
                           gy, finit, fx, fy, stx, sty,
                           stmin, stmax, width, width1))
                #return stp, f, g
                return stp

#              A modified function is used to predict the step during the
#              first stage if a lower function value has been obtained but
#              the decrease is not sufficient.
            #if stage == 1 and f <= fx and f > ftest:
#           #    Define the modified function and derivative values.
            #    fm =f - stp * gtest
            #    fxm = fx - stx * gtest
            #    fym = fy - sty * gtest
            #    gm = g - gtest
            #    gxm = gx - gtest
            #    gym = gy - gtest

#               Call step to update stx, sty, and to compute the new step.
            #    stx, sty, stp, gxm, fxm, gym, fym = self.update (stx, fxm, gxm, sty,
            #                                        fym, gym, stp, fm, gm,
            #                                        stmin, stmax)

#           #    Reset the function and derivative values for f.

            #    fx = fxm + stx * gtest
            #    fy = fym + sty * gtest
            #    gx = gxm + gtest
            #    gy = gym + gtest

            #else:
#           Call step to update stx, sty, and to compute the new step.

            stx, sty, stp, gx, fx, gy, fy= self.update(stx, fx, gx, sty,
                                                       fy, gy, stp, f, g,
                                                       stmin, stmax,
                                                       old_stp, pk)


#           Decide if a bisection step is needed.

            if self.bracket:
                if abs(sty-stx) >= .66 * width1:
                    stp = stx + .5 * (sty - stx)
                width1 = width
                width = abs(sty - stx)

#           Set the minimum and maximum steps allowed for stp.

            if self.bracket:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stp + self.xtrapl * (stp - stx)
                stmax = stp + self.xtrapu * (stp - stx)

#           Force the step to be within the bounds maxstep and minstep.

            stp = max(stp, self.stpmin)
            stp = min(stp, self.stpmax)

            if (stx == stp and stp == self.stpmax and stmin > self.stpmax):
                self.no_update = True
#           If further progress is not possible, let stp be the best
#           point obtained during the search.

            if (self.bracket and stp < stmin or stp >= stmax) \
               or (self.bracket and stmax - stmin < self.xtol * stmax):
                stp = stx

#           Obtain another function and derivative.

            self.tasks[at_idx] = 'FG'
            self.save(at_idx, (stage, ginit, gtest, gx,
                       gy, finit, fx, fy, stx, sty,
                       stmin, stmax, width, width1))
            return stp

    def update(self, stx, fx, gx, sty, fy, gy, stp, fp, gp,
               stpmin, stpmax, old_stp, pk):
        sign = gp * (gx / abs(gx))

#       First case: A higher function value. The minimum is bracketed.
#       If the cubic step is closer to stx than the quadratic step, the
#       cubic step is taken, otherwise the average of the cubic and
#       quadratic steps is taken.
        if fp > fx:  #case1
            self.case = 1
            theta = 3. * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt((theta / s) ** 2. - (gx / s) * (gp / s))
            if stp < stx:
                gamma = -gamma
            p = (gamma - gx) + theta
            q = ((gamma - gx) + gamma) + gp
            r = p / q
            stpc = stx + r * (stp - stx)
            stpq = stx + ((gx / ((fx - fp) / (stp-stx) + gx)) / 2.) \
                   * (stp - stx)
            if (abs(stpc - stx) < abs(stpq - stx)):
               stpf = stpc
            else:
               stpf = stpc + (stpq - stpc) / 2.

            self.bracket = True

#       Second case: A lower function value and derivatives of opposite
#       sign. The minimum is bracketed. If the cubic step is farther from
#       stp than the secant step, the cubic step is taken, otherwise the
#       secant step is taken.

        elif sign < 0:  #case2
            self.case = 2
            theta = 3. * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt((theta / s) ** 2 - (gx / s) * (gp / s))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = ((gamma - gp) + gamma) + gx
            r = p / q
            stpc = stp + r * (stx - stp)
            stpq = stp + (gp / (gp - gx)) * (stx - stp)
            if (abs(stpc - stp) > abs(stpq - stp)):
                stpf = stpc
            else:
                stpf = stpq
            #print (theta, s, gamma, p, q, stpc, stpq, stpf)
            self.bracket = True


#       Third case: A lower function value, derivatives of the same sign,
#       and the magnitude of the derivative decreases.

        elif abs(gp) < abs(gx):  #case3
            self.case = 3
#           The cubic step is computed only if the cubic tends to infinity
#           in the direction of the step or if the minimum of the cubic
#           is beyond stp. Otherwise the cubic step is defined to be the
#           secant step.

            theta = 3. * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))

#           The case gamma = 0 only arises if the cubic does not tend
#           to infinity in the direction of the step.
            gamma = s * np.sqrt(max(0.,(theta / s) ** 2-(gx / s) * (gp / s)))
            if stp > stx:
                gamma = -gamma
            p = (gamma - gp) + theta
            q = (gamma + (gx - gp)) + gamma
            r = p / q
            if r < 0. and gamma != 0:
               stpc = stp + r * (stx - stp)
            elif stp > stx:
               stpc = stpmax
            else:
               stpc = stpmin
            stpq = stp + (gp / (gp - gx)) * (stx - stp)

            if self.bracket:

#               A minimizer has been bracketed. If the cubic step is
#               closer to stp than the secant step, the cubic step is
#               taken, otherwise the secant step is taken.

                if abs(stpc - stp) < abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                if stp > stx:
                    stpf = min(stp + .66 * (sty - stp), stpf)
                else:
                    stpf = max(stp + .66 * (sty - stp), stpf)
            else:

#               A minimizer has not been bracketed. If the cubic step is
#               farther from stp than the secant step, the cubic step is
#               taken, otherwise the secant step is taken.

                if abs(stpc - stp) > abs(stpq - stp):
                   stpf = stpc
                else:
                   stpf = stpq
                stpf = min(stpmax, stpf)
                stpf = max(stpmin, stpf)

#       Fourth case: A lower function value, derivatives of the same sign,
#       and the magnitude of the derivative does not decrease. If the
#       minimum is not bracketed, the step is either minstep or maxstep,
#       otherwise the cubic step is taken.

        else:  #case4
            self.case = 4
            if self.bracket:
                theta = 3. * (fp - fy) / (sty - stp) + gy + gp
                s = max(abs(theta), abs(gy), abs(gp))
                gamma = s * np.sqrt((theta / s) ** 2 - (gy / s) * (gp / s))
                if stp > sty:
                    gamma = -gamma
                p = (gamma - gp) + theta
                q = ((gamma - gp) + gamma) + gy
                r = p / q
                stpc = stp + r * (sty - stp)
                stpf = stpc
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin

#       Update the interval which contains a minimizer.

        if fp > fx:
            sty = stp
            fy = fp
            gy = gp
        else:
            if sign < 0:
                sty = stx
                fy = fx
                gy = gx
            stx = stp
            fx = fp
            gx = gp
#       Compute the new step.

        stp = self.determine_step(stpf, old_stp, pk)

        return stx, sty, stp, gx, fx, gy, fy

    def determine_step(self, stp, old_stp, pk):
        dr = stp - old_stp
        x = np.reshape(pk, (-1, 3))
        steplengths = ((dr*x)**2).sum(1)**0.5
        maxsteplength = pymax(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength
        stp = old_stp + dr
        return stp

    def determine_step_(self, pk):
        x = np.reshape(pk, (-1, 3))
        steplengths = ((x)**2).sum(1)**0.5
        maxsteplength = pymax(steplengths)
        if maxsteplength >= self.maxstep:
            return self.maxstep / maxsteplength
        else:
            return 1.
        

    def save(self, at_idx, data):
        if self.bracket:
            self.isave[at_idx][0] = 1
        else:
            self.isave[at_idx][0] = 0
        self.isave[at_idx][1] = data[0]
        self.dsave[at_idx] = data[1:]

def atoms_list_to_PYG(ase_atoms_list, device):
    data = []
    for ase_atoms in ase_atoms_list:
        z = torch.from_numpy(ase_atoms.numbers).long()
        positions = torch.from_numpy(ase_atoms.positions).float()
        data.append(Data(z=z, pos=positions))
    batch = Batch.from_data_list(data).to(device)
    return batch.pos, batch.z, batch.batch


class AtomsConverterError(Exception):
    pass


class AtomsConverter:
    """
    Convert ASE atoms to SchNetPack input batch format for model prediction.

    """

    def __init__(
        self,
        neighbor_list: Union[schnetpack.transform.Transform, None],
        transforms: Union[
            schnetpack.transform.Transform, List[schnetpack.transform.Transform]
        ] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        additional_inputs: Dict[str, torch.Tensor] = None,
    ):
        """
        Args:
            neighbor_list (schnetpack.transform.Transform, None): neighbor list transform. Can be set to None incase
                that the neighbor list is contained in transforms.
            transforms: transforms for manipulating the neighbor lists. This can be either a single transform or a list
                of transforms that will be executed after the neighbor list is calculated. Such transforms may be
                useful, e.g., for filtering out certain neighbors. In case transforms are required before the neighbor
                list is calculated, neighbor_list argument can be set to None and a list of transforms including the
                neighbor list can be passed as transform argument. The transforms will be executed in the order of
                their appearance in the list.
            device (str, torch.device): device on which the model operates (default: cpu).
            dtype (torch.dtype): required data type for the model input (default: torch.float32).
            additional_inputs (dict): additional inputs required for some transforms.
                When setting up the AtomsConverter, those additional inputs will be
                stored to the input batch.
        """

        self.neighbor_list = deepcopy(neighbor_list)
        self.device = device
        self.dtype = dtype
        self.additional_inputs = additional_inputs or {}

        # convert transforms and neighbor_list to list
        transforms = transforms or []
        if type(transforms) != list:
            transforms = [transforms]
        neighbor_list = [] if neighbor_list is None else [neighbor_list]

        # get transforms and initialize neighbor list
        self.transforms: List[schnetpack.transform.Transform] = (
            neighbor_list + transforms
        )

        # Set numerical precision
        if dtype == torch.float32:
            self.transforms.append(CastTo32())
        elif dtype == torch.float64:
            self.transforms.append(CastTo64())
        else:
            raise AtomsConverterError(f"Unrecognized precision {dtype}")

    def __call__(self, atoms: List[Atoms] or Atoms):
        """

        Args:
            atoms (list or ase.Atoms): list of ASE atoms objects or single ASE atoms object.

        Returns:
            dict[str, torch.Tensor]: input batch for model.
        """

        # check input type and prepare for conversion
        if type(atoms) == list:
            pass
        elif type(atoms) == ase.Atoms:
            atoms = [atoms]
        else:
            raise TypeError(
                "atoms is type {}, but should be either list or ase.Atoms object".format(
                    type(atoms)
                )
            )

        inputs_batch = []
        for at_idx, at in enumerate(atoms):

            inputs = {
                properties.n_atoms: torch.tensor([at.get_global_number_of_atoms()]),
                properties.Z: torch.from_numpy(at.get_atomic_numbers()),
                properties.R: torch.from_numpy(at.get_positions()),
                properties.cell: torch.from_numpy(at.get_cell().array).view(-1, 3, 3),
                properties.pbc: torch.from_numpy(at.get_pbc()).view(-1, 3),
            }

            # specify sample index
            inputs.update({properties.idx: torch.tensor([at_idx])})

            # add additional inputs (specified in AtomsConverter __init__)
            inputs.update(self.additional_inputs)

            for transform in self.transforms:
                inputs = transform(inputs)
            inputs_batch.append(inputs)

        inputs = _atoms_collate_fn(inputs_batch)

        # Move input batch to device
        inputs = {p: inputs[p].to(self.device) for p in inputs}

        return inputs


class BatchwiseCalculator:
    """
    Calculator for neural network models for batchwise optimization.
    """

    def __init__(
        self,
        model: nn.Module or str,
        atoms_converter: AtomsConverter,
        device: str or torch.device = "cpu",
        energy_key: str = "energy",
        force_key: str = "forces",
        energy_unit: str = "eV",
        position_unit: str = "Ang",
        dtype: torch.dtype = torch.float32,
    ):
        """
        model:
            path to trained model or trained model

        atoms_converter:
            Class used to convert ase Atoms objects to schnetpack input

        device:
            device used for calculations (default="cpu")

        auxiliary_output_modules:
            auxiliary module to manipulate output properties (e.g., prior energy or forces)

        energy_key:
            name of energies in model (default="energy")

        force_key:
            name of forces in model (default="forces")

        energy_unit:
            energy units used by model (default="eV")

        position_unit:
            position units used by model (default="Angstrom")

        dtype:
            required data type for the model input (default: torch.float32)
        """

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

        # load model from path if needed
        self.model = model
        self.model.to(device=self.device, dtype=self.dtype)
        self.atoms_converter = atoms_converter

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

    def get_forces(self, atoms: List[ase.Atoms], fixed_atoms_mask: Optional[List[int]] = None) -> np.array:
        """
        atoms:

        fixed_atoms_mask:
            list of indices corresponding to atoms with positions fixed in space.
        """
        if self._requires_calculation(property_keys=[self.energy_key, self.force_key], atoms=atoms):
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


class BatchwiseDynamics(Dynamics):
    """Base-class for batch-wise MD and structure optimization classes."""
    atoms = None
    def __init__(
        self,
        calculator: BatchwiseCalculator,
        logfile: str,
        trajectory: Optional[str],
        append_trajectory: bool = False,
        master: Optional[bool] = None,
        log_every_step: bool = False,
        fixed_atoms_mask: Optional[List[int]]=None,
    ):
        """Structure dynamics object.

        Parameters:

        calculator:
            This calculator provides properties such as forces and energy, which can be used for MD simulations or
            relaxations

        atoms:
            The Atoms objects to relax.

        restart:
            Filename for restart file.  Default value is *None*.

        logfile:
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory:
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        append_trajectory:
            Appended to the trajectory file instead of overwriting it.

        master:
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        log_every_step:
            set to True to log Dynamics after each step (default=False)

        fixed_atoms:
            list of indices corresponding to atoms with positions fixed in space.
        """
        super().__init__(
            atoms=None,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            master=master,
        )

        self.calculator = calculator
        self.trajectory = trajectory
        self.log_every_step = log_every_step
        self.fixed_atoms_mask = fixed_atoms_mask
        #self.n_configs = len(self.atoms)
        #self.n_atoms = len(self.atoms[0])

    def irun(self):
        # compute initial structure and log the first step
        self.calculator.get_forces(self.atoms, fixed_atoms_mask=self.fixed_atoms_mask)

        # yield the first time to inspect before logging
        yield False

        if self.nsteps == 0:
            self.log()
            pass

        # run the algorithm until converged or max_steps reached
        while not self.converged() and self.nsteps < self.max_steps:

            # compute the next step
            self.step()
            self.nsteps += 1

            # let the user inspect the step and change things before logging
            # and predicting the next step
            yield False

            # log the step
            if self.log_every_step:
                self.log()

        # log last step
        self.log()

        # finally check if algorithm was converged
        yield self.converged()

    def run(self) -> bool:
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        for converged in BatchwiseDynamics.irun(self):
            pass
        return converged


class BatchwiseOptimizer(BatchwiseDynamics):
    """Base-class for all structure optimization classes."""

    # default maxstep for all optimizers
    defaults = {"maxstep": 0.2}

    def __init__(
        self,
        calculator: BatchwiseCalculator,
        restart: Optional[bool] = None,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        master: Optional[str] = None,
        append_trajectory: bool = False,
        log_every_step: bool = False,
        fixed_atoms_mask: Optional[List[int]] = None,
    ):
        """Structure optimizer object.

        Parameters:

        calculator:
            This calculator provides properties such as forces and energy, which can be used for MD simulations or
            relaxations

        atoms:
            The Atoms objects to relax.

        restart:
            Filename for restart file.  Default value is *None*.

        logfile:
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory:
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        master:
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        append_trajectory:
            Appended to the trajectory file instead of overwriting it.

        log_every_step:
            set to True to log Dynamics after each step (default=False)

        fixed_atoms:
            list of indices corresponding to atoms with positions fixed in space.
        """
        BatchwiseDynamics.__init__(
            self,
            calculator=calculator,
            logfile=logfile,
            trajectory=trajectory,
            master=master,
            append_trajectory=append_trajectory,
            log_every_step=log_every_step,
            fixed_atoms_mask=fixed_atoms_mask,
        )

        self.restart = restart

        # initialize attribute
        self.fmax = None

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            barrier()

    def todict(self) -> Dict:
        description = {
            "type": "optimization",
            "optimizer": self.__class__.__name__,
        }
        return description

    def initialize(self):
        pass

    def irun(self, atoms, fmax: float = 0.05, steps: Optional[int] = None):
        """call Dynamics.irun and keep track of fmax"""
        self.atoms = atoms
        self.n_configs = len(atoms)
        self.n_ats = sum([len(at.numbers) for at in self.atoms])
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return BatchwiseDynamics.irun(self)

    def run(self, atoms, fmax: float = 0.05, steps: Optional[int] = None):
        """call Dynamics.run and keep track of fmax"""
        self.atoms = atoms
        self.n_configs = len(atoms)
        self.n_ats = sum([len(at.numbers) for at in self.atoms])
        self.n_ats_per_config = np.array([len(at.numbers) for at in self.atoms])
        self.ats_mask = np.zeros((self.n_configs, self.n_ats), dtype=bool)
        self.batch = np.array(sum([[i for _ in range(len(self.atoms[i].numbers))] for i in range(self.n_configs)], []))
        current_pos = 0
        for config_idx, at in enumerate(self.atoms):
            first_idx = current_pos
            last_idx = current_pos + len(at.numbers)
            current_pos = last_idx
            self.ats_mask[config_idx, first_idx:last_idx] = True
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return BatchwiseDynamics.run(self)

    def converged(self, forces: Optional[np.array] = None) -> bool:
        """Did the optimization converge?"""
        if forces is None:
            forces = self.calculator.get_forces(
                self.atoms, fixed_atoms_mask=self.fixed_atoms_mask
            )
        # todo: maybe np.linalg.norm?
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def log(self, forces: Optional[np.array] = None) -> None:
        if forces is None:
            forces = self.calculator.get_forces(
                self.atoms, fixed_atoms_mask=self.fixed_atoms_mask
            )
        fmax = sqrt((forces**2).sum(axis=1).max())
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "fmax")
                msg = "%s  %4s %8s %12s\n" % args
                self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], fmax)
            msg = "%s:  %3d %02d:%02d:%02d %12.4f\n" % args
            self.logfile.write(msg)

            self.logfile.flush()

        if self.trajectory is not None:
            for struc_idx, at in enumerate(self.atoms):
                # store in trajectory
                write(
                    self.trajectory + "_{}.xyz".format(struc_idx),
                    at,
                    format="extxyz",
                    append=False if self.nsteps == 0 else True,
                )

    def get_relaxation_results(self) -> Tuple[Atoms, Dict]:
        self.calculator.get_forces(self.atoms)
        return self.atoms, self.calculator.results

    def dump(self, data):
        if world.rank == 0 and self.restart is not None:
            with open(self.restart, "wb") as fd:
                pickle.dump(data, fd, protocol=2)

    def load(self):
        with open(self.restart, "rb") as fd:
            return pickle.load(fd)


class ASEBatchwiseLBFGS(BatchwiseOptimizer):
    """Limited memory BFGS optimizer.

    LBFGS optimizer that allows for relaxation of multiple structures in parallel. This optimizer is an
    extension/adaptation of the ase.optimize.LBFGS optimizer particularly designed for batch-wise relaxation
    of atomic structures. The inverse Hessian is approximated for each sample separately, which allows for
    optimizing batches of different structures/compositions.

    """
    atoms = None
    def __init__(
        self,
        calculator: BatchwiseCalculator,
        restart: Optional[bool] = None,
        logfile: str = "-",
        trajectory: Optional[str] = None,
        maxstep: Optional[float] = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        use_line_search: bool = False,
        master: Optional[str] = None,
        log_every_step: bool = False,
        fixed_atoms_mask: Optional[List[int]] = None,
        verbose: bool = False,
    ):

        """Parameters:

        calculator:
            This calculator provides properties such as forces and energy, which can be used for MD simulations or
            relaxations

        atoms:
            The Atoms objects to relax.

        restart:
            Pickle file used to store vectors for updating the inverse of
            Hessian matrix. If set, file with such a name will be searched
            and information stored will be used, if the file exists.

        logfile:
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory:
            Pickle file used to store trajectory of atomic movement.

        maxstep:
            How far is a single atom allowed to move. This is useful for DFT
            calculations where wavefunctions can be reused if steps are small.
            Default is 0.2 Angstrom.

        memory:
            Number of steps to be stored. Default value is 100. Three numpy
            arrays of this length containing floats are stored.

        damping:
            The calculated step is multiplied with this number before added to
            the positions.

        alpha:
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        use_line_search:
            Not implemented yet.

        master:
            Defaults to None, which causes only rank 0 to save files.  If
            set to true, this rank will save files.

        log_every_step:
            set to True to log Dynamics after each step (default=False)

        fixed_atoms:
            list of indices corresponding to atoms with positions fixed in space.
        """

        BatchwiseOptimizer.__init__(
            self,
            calculator=calculator,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            master=master,
            log_every_step=log_every_step,
            fixed_atoms_mask=fixed_atoms_mask,
        )

        if maxstep is not None:
            self.maxstep = maxstep
        else:
            self.maxstep = self.defaults["maxstep"]

        if self.maxstep > 1.0:
            raise ValueError(
                "You are using a much too large value for "
                + "the maximum step size: %.1f Angstrom" % maxstep
            )

        self.memory = memory
        # Initial approximation of inverse Hessian 1./70. is to emulate the
        # behaviour of BFGS. Note that this is never changed!
        self.H0 = 1.0 / alpha
        self.damping = damping
        self.use_line_search = use_line_search
        self.p = None
        self.function_calls = 0
        self.force_calls = 0
        self.n_normalizations = 0

        self.verbose = verbose

        #if use_line_search:
        #    raise NotImplementedError("Lines search has not been implemented yet")

    def initialize(self) -> None:
        """Initialize everything so no checks have to be done in step"""
        self.iteration = 0
        self.s = []
        self.y = []
        # Store also rho, to avoid calculating the dot product again and
        # again.
        self.rho = []

        self.r0 = None
        self.f0 = None
        self.e0 = None
        self.task = "START"
        self.load_restart = False

    def read(self) -> None:
        """Load saved arrays to reconstruct the Hessian"""
        (
            self.iteration,
            self.s,
            self.y,
            self.rho,
            self.r0,
            self.f0,
            self.e0,
            self.task,
        ) = self.load()
        self.load_restart = True

    def step(self, f: np.array = None) -> None:
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it"""

        if f is None:
            f = self.calculator.get_forces(
                self.atoms, fixed_atoms_mask=self.fixed_atoms_mask
            )
        r = np.zeros((self.n_ats, 3), dtype=np.float64)
        current_pos = 0
        mask = []
        for config_idx, at in enumerate(self.atoms):
            first_idx = current_pos
            last_idx = current_pos + len(at.numbers)
            current_pos = last_idx
            r[first_idx:last_idx] = at.get_positions()
            q_euclidean = -f[first_idx:last_idx].reshape(-1, 3)
            squared_max_forces = (q_euclidean**2).sum(axis=-1).max(axis=-1)
            mask.append( np.array([squared_max_forces < self.fmax**2])[:, None].repeat(3, 1).repeat(last_idx - first_idx, 0))
        # check if updates for respective structures are required
        #q_euclidean = -f.reshape(self.n_configs, -1, 3)
        #squared_max_forces = (q_euclidean**2).sum(axis=-1).max(axis=-1)
        #for config_idx, at in enumerate(self.atoms):

        #configs_mask = squared_max_forces < self.fmax**2
        #mask = (
        #    configs_mask[:, None]
        #    .repeat(q_euclidean.shape[1], 0)
        #    .repeat(q_euclidean.shape[2], 1)
        #)
        mask = np.concatenate(mask, axis=0)
        self.update(r, f, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty(
            (
                loopmax,
                self.n_ats,
                1,
            ),
            dtype=np.float64,
        )
        b =  np.empty(
            (
                self.n_ats,
                1
            ),
            dtype=np.float64,
        )

        # ## The algorithm itself:
        q = -f
        for i in range(loopmax - 1, -1, -1):
            #for at_idx in range(self.n_configs):
            # print (rho[i].shape)
            #    a[i][self.ats_mask[at_idx]] = rho[i][at_idx] * s[i][self.ats_mask[at_idx]].reshape(-1).dot(q[self.ats_mask[at_idx]].reshape(-1)) 
             
            per_atom_ai = rho[i] * np_scatter_add( (s[i].reshape(-1, 1) * q.reshape(-1, 1)).reshape(-1, 3), self.batch, (self.n_configs, 3)).sum(axis=1)
            a[i] = np.repeat(per_atom_ai, self.n_ats_per_config, axis=0).reshape(-1, 1)
            q -= a[i]  * y[i]

        z = H0 * q

        for i in range(loopmax):
            b = rho[i] * np_scatter_add((y[i].reshape(-1, 1) * z.reshape(-1, 1)).reshape(-1, 3), self.batch, (self.n_configs, 3)).sum(axis=1)
            b = np.repeat(b, self.n_ats_per_config, axis=0).reshape(-1, 1)
            #for at_idx in range(self.n_configs):
            #   b[self.ats_mask[at_idx]] = rho[i][at_idx] * y[i][self.ats_mask[at_idx]].reshape(-1).dot(z[self.ats_mask[at_idx]].reshape(-1))

            z += s[i] * (a[i] - b)

        p = -z.reshape((-1, 3))
        self.p = np.where(mask, np.zeros_like(p), p)
        # ##

        g = -f
        if self.use_line_search is True:
            e = self.func(r)
            # default_step = self.determine_step(self.p)
            self.line_search(r, g, e)# , default_step)

            dr = (self.p * self.alpha_k.reshape(self.n_ats, -1)).reshape(r.shape[0], -1)
        else:
            self.force_calls += 1
            self.function_calls += 1
            dr = self.determine_step(self.p) * self.damping

        # update positions
        pos_updated = r + dr

        # create new list of ase Atoms objects with updated positions
        ats = []
        current_pos = 0
        for config_idx, at in enumerate(self.atoms):
            first_idx = current_pos
            last_idx = current_pos + len(at.numbers)
            current_pos = last_idx
            at = Atoms(
                positions=pos_updated[first_idx:last_idx],
                numbers=self.atoms[config_idx].get_atomic_numbers(),
            )
            at.pbc = self.atoms[config_idx].pbc
            at.cell = self.atoms[config_idx].cell
            ats.append(at)
        self.atoms = ats

        self.iteration += 1
        self.r0 = r
        self.f0 = -g
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )

    def determine_step(self, dr: np.array) -> np.array:
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        steplengths = (dr**2).sum(-1) ** 0.5
        # check if any step in entire batch is greater than maxstep
        if np.max(steplengths) >= self.maxstep:
            # rescale steps for each config separately
            current_pos = 0
            for config_idx, at in enumerate(self.atoms):
                first_idx = current_pos
                last_idx = current_pos + len(at.numbers)
                current_pos = last_idx
                longest_step = np.max(steplengths[first_idx:last_idx])
                if longest_step >= self.maxstep:
                    if self.verbose:
                        print("normalized integration step")
                    self.n_normalizations += 1
                    dr[first_idx:last_idx] *= self.maxstep / longest_step
        return dr

    def update(self, r: np.array, f: np.array, r0: np.array, f0: np.array) -> None:
        """Update everything that is kept in memory

        This function is mostly here to allow for replay_trajectory.
        """
        if self.iteration > 0:
            s0 = r - r0
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0 - f
            self.y.append(y0)

            rho0 = np.ones((self.n_configs), dtype=np.float64)
            for config_idx in range(self.n_configs):
                ys0 = np.dot(y0[self.ats_mask[config_idx]].reshape(-1), s0[self.ats_mask[config_idx]].reshape(-1))
                if ys0 > 1e-8:
                    rho0[config_idx] = 1.0 / ys0
            self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def _set_positions(self, x):
        ats = []
        x = x.reshape(-1, 3)
        current_pos = 0
        for config_idx, at in enumerate(self.atoms):
            first_idx = current_pos
            last_idx = current_pos + len(at.numbers)
            current_pos = last_idx
            at = Atoms(
                positions=x[first_idx:last_idx],
                numbers=self.atoms[config_idx].get_atomic_numbers(),
            )
            at.pbc = self.atoms[config_idx].pbc
            at.cell = self.atoms[config_idx].cell
            ats.append(at)
        self.atoms = ats    

    def func(self, x):
        """Objective function for use of the optimizers"""
        self._set_positions(x)
        self.function_calls += 1
        return self.calculator.get_potential_energy(self.atoms)
    
    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self._set_positions(x)
        self.force_calls += 1
        # Remember that forces are minus the gradient!
        return - self.calculator.get_forces(self.atoms)

    def line_search(self, r, g, e):#, default_step=1.):
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = \
            ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0,
                            self.n_configs, self.ats_mask, self.batch, self.n_ats_per_config, self.n_ats,
                            maxstep=self.maxstep, c1=.23, c2=.46, stpmax=50.)
        if self.alpha_k is None:
            raise RuntimeError('LineSearch failed!')
