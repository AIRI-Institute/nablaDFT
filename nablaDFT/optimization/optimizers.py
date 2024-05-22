from typing import Dict, Optional, List, Tuple
from os.path import isfile
import pickle
import time
from math import sqrt

import numpy as np
from ase.optimize.optimize import Dynamics
from ase.parallel import world, barrier
from ase.io import write
from ase import Atoms

from .line_search import LineSearch
from .opt_utils import np_scatter_add
from .calculator import BatchwiseCalculator


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
        fixed_atoms_mask: Optional[List[int]] = None,
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
        self.batch = np.array(
            sum(
                [
                    [i for _ in range(len(self.atoms[i].numbers))]
                    for i in range(self.n_configs)
                ],
                [],
            )
        )
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
        """Args:

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
            mask.append(
                np.array([squared_max_forces < self.fmax**2])[:, None]
                .repeat(3, 1)
                .repeat(last_idx - first_idx, 0)
            )
        # check if updates for respective structures are required
        # q_euclidean = -f.reshape(self.n_configs, -1, 3)
        # squared_max_forces = (q_euclidean**2).sum(axis=-1).max(axis=-1)
        # for config_idx, at in enumerate(self.atoms):

        # configs_mask = squared_max_forces < self.fmax**2
        # mask = (
        #    configs_mask[:, None]
        #    .repeat(q_euclidean.shape[1], 0)
        #    .repeat(q_euclidean.shape[2], 1)
        # )
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
        b = np.empty(
            (self.n_ats, 1),
            dtype=np.float64,
        )

        # ## The algorithm itself:
        q = -f
        for i in range(loopmax - 1, -1, -1):
            # for at_idx in range(self.n_configs):
            # print (rho[i].shape)
            #    a[i][self.ats_mask[at_idx]] = rho[i][at_idx] * s[i][self.ats_mask[at_idx]].reshape(-1).dot(q[self.ats_mask[at_idx]].reshape(-1))

            per_atom_ai = rho[i] * np_scatter_add(
                (s[i].reshape(-1, 1) * q.reshape(-1, 1)).reshape(-1, 3),
                self.batch,
                (self.n_configs, 3),
            ).sum(axis=1)
            a[i] = np.repeat(per_atom_ai, self.n_ats_per_config, axis=0).reshape(-1, 1)
            q -= a[i] * y[i]

        z = H0 * q

        for i in range(loopmax):
            b = rho[i] * np_scatter_add(
                (y[i].reshape(-1, 1) * z.reshape(-1, 1)).reshape(-1, 3),
                self.batch,
                (self.n_configs, 3),
            ).sum(axis=1)
            b = np.repeat(b, self.n_ats_per_config, axis=0).reshape(-1, 1)
            # for at_idx in range(self.n_configs):
            #   b[self.ats_mask[at_idx]] = rho[i][at_idx] * y[i][self.ats_mask[at_idx]].reshape(-1).dot(z[self.ats_mask[at_idx]].reshape(-1))

            z += s[i] * (a[i] - b)

        p = -z.reshape((-1, 3))
        self.p = np.where(mask, np.zeros_like(p), p)
        # ##

        g = -f
        if self.use_line_search is True:
            e = self.func(r)
            self.line_search(r, g, e)

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
                ys0 = np.dot(
                    y0[self.ats_mask[config_idx]].reshape(-1),
                    s0[self.ats_mask[config_idx]].reshape(-1),
                )
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
        return -self.calculator.get_forces(self.atoms)

    def line_search(self, r, g, e):  # , default_step=1.):
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = ls._line_search(
            self.func,
            self.fprime,
            r,
            self.p,
            g,
            e,
            self.e0,
            self.n_configs,
            self.ats_mask,
            self.batch,
            self.n_ats_per_config,
            self.n_ats,
            maxstep=self.maxstep,
            c1=0.23,
            c2=0.46,
            stpmax=50.0,
        )
        if self.alpha_k is None:
            raise RuntimeError("LineSearch failed!")
