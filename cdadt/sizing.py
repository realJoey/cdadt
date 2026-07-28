"""The sizing analysis: build the coupled model, converge it, read the results.

This is the class that treats OpenConcept's mission as a black box. It sets the inputs --
design parameters, mission profile, solver settings -- converges the coupled system, and reads
a fixed, named set of outputs back. Nothing in cdadt reaches into the mission's internals to
recompute a quantity OpenConcept already produces, and nothing modifies OpenConcept.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
import openmdao.api as om

from cdadt.aircraft import AircraftDefinition
from cdadt.mission import MissionProfile
from cdadt.model import JetTransportPhaseModel, SizingModel

__all__ = ["SizingAnalysis"]


class SizingAnalysis:
    """A full mission sizing analysis of one aircraft against one mission.

    Parameters
    ----------
    aircraft : AircraftDefinition
        Design parameters.
    profile : MissionProfile
        Mission to fly and the continuation schedule that converges it.
    phase_model : type, optional
        Aircraft model class OpenConcept builds inside each phase. Default
        :class:`~cdadt.model.JetTransportPhaseModel`.
    num_nodes : int, optional
        Analysis points per phase; must be odd. Default 11. OpenConcept's own B738 sizing
        example runs 21, and cdadt's reference validation is run at 21.
    maxiter, atol, rtol : optional
        Newton solver settings. Defaults 20, 1e-9, 1e-9, matching the reference example.
    iprint : int, optional
        Solver print level. Default -1, silent.
    err_on_non_converge : bool, optional
        Raise instead of returning the state of a failed solve. Default ``True``. A
        non-converged mission still produces numbers -- a negative field length, a range that
        misses the one requested -- and they are indistinguishable from results.
    initial_MTOW, initial_fuel : float, optional
        Starting guesses in kg for the two states that drive the coupled solve, 50e3 and 30e3.

    Examples
    --------
    >>> analysis = SizingAnalysis(aircraft, profile, num_nodes=21)
    >>> results = analysis.run()
    >>> results["MTOW"]
    78345.6...
    """

    #: Every result this analysis reports: ``name -> (problem path, units)``. Reading the whole
    #: set rather than whichever few a caller asked for is what makes a run report complete.
    RESULTS: ClassVar[dict[str, tuple[str, str | None]]] = {
        "MTOW": ("ac|weights|MTOW", "kg"),
        "OEW": ("ac|weights|OEW", "kg"),
        "MLW": ("ac|weights|MLW", "kg"),
        "payload": ("ac|weights|W_payload", "kg"),
        "block_fuel": ("mission.descent.fuel_burn_final", "kg"),
        "total_fuel": ("mission.loiter.fuel_burn_final", "kg"),
        "takeoff_field_length": ("mission.bfl.distance_continue", "ft"),
        "abort_distance": ("mission.bfl.distance_abort", "ft"),
        "V1": ("mission.takeoff|v1", "kn"),
        "V2": ("mission.engineoutclimb.takeoff|v2", "kn"),
        "engine_out_climb_angle": ("mission.engineoutclimb.gamma", "rad"),
        "CLmax_cruise": ("ac|aero|CLmax_cruise", None),
        "CLmax_TO": ("ac|aero|CLmax_TO", None),
        "wing_MAC": ("ac|geom|wing|MAC", "m"),
        "hstab_S_ref": ("ac|geom|hstab|S_ref", "m**2"),
        "vstab_S_ref": ("ac|geom|vstab|S_ref", "m**2"),
        "wing_S_ref": ("ac|geom|wing|S_ref", "m**2"),
        "wing_AR": ("ac|geom|wing|AR", None),
        "engine_rating": ("ac|propulsion|engine|rating", "lbf"),
    }

    def __init__(
        self,
        aircraft: AircraftDefinition,
        profile: MissionProfile,
        phase_model: type = JetTransportPhaseModel,
        num_nodes: int = 11,
        maxiter: int = 20,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        iprint: int = -1,
        err_on_non_converge: bool = True,
        initial_MTOW: float = 50e3,
        initial_fuel: float = 30e3,
    ) -> None:
        self.aircraft = aircraft
        self.profile = profile
        self.phase_model = phase_model
        self.num_nodes = num_nodes
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.iprint = iprint
        self.err_on_non_converge = err_on_non_converge
        self.initial_MTOW = initial_MTOW
        self.initial_fuel = initial_fuel
        self._problem: om.Problem | None = None

    # -- building -----------------------------------------------------------------------

    @property
    def problem(self) -> om.Problem:
        """Return the built problem.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called.
        """
        if self._problem is None:
            raise RuntimeError("This analysis has not been built yet; call build() or run().")
        return self._problem

    def build(self, hooks: Sequence[Callable[[om.Group], None]] = ()) -> om.Problem:
        """Assemble and set up the problem.

        Parameters
        ----------
        hooks : sequence of callable, optional
            Called with the model after it is assembled and before ``setup``. This is where
            design variables, an objective and constraints are registered, because OpenMDAO
            requires all three before setup. :class:`~cdadt.optimization.DesignOptimizer`
            supplies one.

        Returns
        -------
        openmdao.api.Problem
            The problem, set up but not yet converged. The mission profile has not been
            applied; call :meth:`converge`.
        """
        problem = om.Problem(
            model=SizingModel(
                aircraft=self.aircraft,
                phase_model=self.phase_model,
                num_nodes=self.num_nodes,
                initial_MTOW=self.initial_MTOW,
                initial_fuel=self.initial_fuel,
            ),
            reports=False,
        )
        self._attach_solvers(problem.model)
        for hook in hooks:
            hook(problem.model)
        problem.setup(check=False)
        self._problem = problem
        return problem

    def _attach_solvers(self, model: om.Group) -> None:
        """Attach the Newton solver that converges the sizing loop and the mission together.

        ``solve_subsystems`` is required rather than optional: the mission's own balances --
        phase durations, throttle, V\\ :sub:`1` -- have to be driven alongside the weight
        closure, not after it. The bounds-enforcing line search keeps the solver inside the
        ranges OpenConcept declares on its implicit states, several of which are nonphysical
        if crossed.
        """
        newton = om.NewtonSolver(solve_subsystems=True)
        newton.options["maxiter"] = self.maxiter
        newton.options["atol"] = self.atol
        newton.options["rtol"] = self.rtol
        newton.options["iprint"] = self.iprint
        newton.options["err_on_non_converge"] = self.err_on_non_converge
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["iprint"] = self.iprint

        model.nonlinear_solver = newton
        model.linear_solver = om.DirectSolver()

    # -- running ------------------------------------------------------------------------

    def converge(self, verbose: bool = False) -> None:
        """Run the continuation schedule and then the design mission.

        Raises
        ------
        openmdao.core.analysis_error.AnalysisError
            If the Newton solver fails and ``err_on_non_converge`` is set.
        """
        self.profile.converge(self.problem, self.num_nodes, SizingModel.MISSION, verbose=verbose)

    def run(self, verbose: bool = False) -> dict[str, float]:
        """Build, converge, and return the results.

        Parameters
        ----------
        verbose : bool, optional
            Print the continuation steps. Default ``False``.

        Returns
        -------
        dict
            Every entry of :attr:`RESULTS`.
        """
        self.build()
        self.converge(verbose=verbose)
        return self.results()

    def results(self) -> dict[str, float]:
        """Return every declared result from the converged problem."""
        problem = self.problem
        return {
            name: float(np.asarray(problem.get_val(path, units=units)).reshape(-1)[0])
            for name, (path, units) in self.RESULTS.items()
        }

    def report(self) -> str:
        """Return the results as a formatted table."""
        units = {name: spec[1] or "-" for name, spec in self.RESULTS.items()}
        lines = [f"{'quantity':<24s} {'value':>16s}  units", "-" * 50]
        for name, value in self.results().items():
            lines.append(f"{name:<24s} {value:16.4f}  {units[name]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation naming the phase model and node count."""
        return f"SizingAnalysis(phase_model={self.phase_model.__name__}, num_nodes={self.num_nodes})"
