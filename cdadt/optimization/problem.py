"""The design problem: the one place a cdadt optimization is assembled.

:class:`DesignProblem` owns discipline assembly, the sizing loop, the certification basis,
the design variables, the objective, and the driver. Nothing else assembles a model. That
centralization is the point: a design study whose problem definition is spread across a run
script is a study whose problem definition nobody can state.

The class refuses to report a non-optimal run as an optimum. An optimizer that hit its
iteration limit still returns a design vector and a full set of results, and those results
are indistinguishable from a converged optimum unless something checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import openmdao.api as om

from cdadt.certification.basis import CertificationBasis
from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.discipline import Discipline
from cdadt.mission.blackbox import MissionProfile
from cdadt.mission.contract import MISSION_OUTPUTS_BY_NAME
from cdadt.mission.sizing import SizingLoop
from cdadt.optimization.driver import OptimizationOutcome, OptimizerDriver

__all__ = ["DesignProblem", "DesignVariable", "OptimizationResult"]


@dataclass(frozen=True)
class DesignVariable:
    """One design variable, as configured.

    Parameters
    ----------
    name : str
        The variable's pipe-separated name, e.g. ``"ac|geom|wing|S_ref"``.
    lower, upper : float
        Bounds, in ``units``.
    units : str or None
        Units the bounds are expressed in.
    scaler : float or None
        Optional scaling applied by the optimizer.
    """

    name: str
    lower: float
    upper: float
    units: str | None
    scaler: float | None = None

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(
                f"Design variable '{self.name}' has lower bound {self.lower} at or above its upper "
                f"bound {self.upper}. An empty range makes the variable silently fixed."
            )

    @property
    def effective_scaler(self) -> float:
        """Return the scaler to give the optimizer.

        The configured value if there is one, otherwise ``1 / max(|lower|, |upper|)`` so the
        variable is order one over its range. Non-dimensionalizing by the bound is a choice
        about the optimizer's numerics, not about the design, which is why it does not need
        to be configured -- but a configured scaler always wins, because a particular
        problem may need different treatment.
        """
        if self.scaler is not None:
            return self.scaler
        magnitude = max(abs(self.lower), abs(self.upper))
        return 1.0 / magnitude if magnitude > 0.0 else 1.0


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of a design optimization.

    Parameters
    ----------
    outcome : OptimizationOutcome
        Whether the optimizer reached an optimum, and how it exited.
    objective_name : str
        The quantity minimized.
    objective_value : float
        Its value at the final point.
    baseline_objective : float
        Its value at the converged baseline, before optimization. Reported so the run says
        what it achieved: an optimizer that converged but improved nothing usually means the
        design variables never reached the model.
    design_variables : dict
        Final value of each design variable, in its configured units.
    results : dict
        Every sizing and mission result at the final point.
    traceability_matrix : str
        The certification traceability matrix at the final point.
    """

    outcome: OptimizationOutcome
    objective_name: str
    objective_value: float
    baseline_objective: float
    design_variables: dict[str, float]
    results: dict[str, float]
    traceability_matrix: str

    @property
    def optimal(self) -> bool:
        """Return whether the optimizer reported convergence to an optimum."""
        return self.outcome.optimal

    @property
    def improvement(self) -> float:
        """Return the fractional reduction in the objective, relative to the baseline."""
        if self.baseline_objective == 0.0:
            return 0.0
        return (self.baseline_objective - self.objective_value) / abs(self.baseline_objective)

    def report(self) -> str:
        """Return a plain-text summary of the run.

        The optimizer's exit status leads, before any numbers. A reader who skims the design
        variables and never sees that the run stopped at its iteration limit has been misled
        by the report's layout rather than by its contents.
        """
        lines = [
            "=" * 78,
            f"cdadt design optimization: minimize {self.objective_name}",
            "=" * 78,
            f"  status:    {self.outcome.status}",
            f"  optimal:   {'YES' if self.optimal else 'NO -- results below are NOT an optimum'}",
            f"  objective: {self.baseline_objective:.4f} -> {self.objective_value:.4f} "
            f"({self.improvement * 100:+.2f}%)",
            "",
            "Design variables",
            "-" * 78,
        ]
        for name, value in sorted(self.design_variables.items()):
            lines.append(f"  {name:36s} {value:14.4f}")

        lines += ["", "Sized aircraft", "-" * 78]
        for name in sorted(self.results):
            lines.append(f"  {name:36s} {self.results[name]:14.4f}")

        lines += ["", "Certification basis", "-" * 78, self.traceability_matrix]
        return "\n".join(lines)


class DesignProblem:
    """A complete cdadt design optimization.

    Parameters
    ----------
    disciplines : sequence of Discipline
        Every discipline in the model, aircraft- and phase-scoped.
    config : AircraftConfiguration
        Configuration for the aircraft, mission, certification basis, solver and optimizer.
    certification : CertificationBasis
        The requirements the design must meet. Required, not optional: a "design
        optimization" with no certification basis is an unconstrained minimization, and
        this class is not for that.
    driver : OptimizerDriver
        The optimizer to run.
    objective : str
        What to minimize. One of the names in
        :data:`~cdadt.mission.contract.MISSION_OUTPUTS`, or a sizing result such as
        ``"MTOW"``. **Required, with no default** -- what a design is being optimized for is
        the single most consequential choice in a study, and inheriting it silently would be
        the worst possible place to save a line.
    profile : MissionProfile, optional
        The mission. Defaults to the one in the configuration.
    num_nodes : int, optional
        Analysis points per mission phase. Default 11.

    Raises
    ------
    ValueError
        If the objective is not a quantity the model produces, or if no design variables are
        configured.
    MissingConfigurationError
        If a design variable's bounds, or an optimizer setting, is absent.
    """

    #: Sizing results usable as objectives, beyond the declared mission outputs.
    SIZING_OBJECTIVES = ("MTOW", "OEW")

    #: Configuration prefix design variables are declared under.
    DESIGN_VARIABLE_ROOT = "optimization|design_variables"

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        config: AircraftConfiguration,
        certification: CertificationBasis,
        driver: OptimizerDriver,
        objective: str,
        profile: MissionProfile | None = None,
        num_nodes: int = 11,
    ) -> None:
        self._config = config
        self._certification = certification
        self._driver = driver
        self._objective = objective
        self._loop = SizingLoop(
            disciplines,
            config,
            profile=profile,
            num_nodes=num_nodes,
            certification=certification,
        )
        self._design_variables = self._read_design_variables()
        self._validate_objective()

    # -- inspection ---------------------------------------------------------------------

    @property
    def sizing_loop(self) -> SizingLoop:
        """Return the sizing loop underneath this optimization."""
        return self._loop

    @property
    def certification(self) -> CertificationBasis:
        """Return the certification basis being optimized against."""
        return self._certification

    @property
    def design_variables(self) -> tuple[DesignVariable, ...]:
        """Return the configured design variables."""
        return self._design_variables

    @property
    def objective(self) -> str:
        """Return the name of the quantity being minimized."""
        return self._objective

    def objective_path(self) -> str:
        """Return the problem path of the objective.

        Returns
        -------
        str
            A path usable with :meth:`openmdao.api.Problem.get_val`.
        """
        if self._objective in self.SIZING_OBJECTIVES:
            return f"ac|weights|{self._objective}"
        return self._loop.blackbox.path(self._objective)

    def _objective_scaler(self) -> float:
        """Return the scaler applied to the objective.

        Read from ``optimization|objective_reference``, which is the order of magnitude the
        objective is expected to take. The objective is divided by it so the optimizer sees
        a quantity near one alongside the non-dimensionalized constraints.

        Raises
        ------
        MissingConfigurationError
            If no reference is configured. There is no sensible default: a fuel burn in
            kilograms and an empty weight in kilograms differ by a factor of three, and a
            takeoff distance in feet by three more.
        """
        self._config.require_all(["optimization|objective_reference"])
        reference = self._config.scalar("optimization|objective_reference")
        if reference <= 0.0:
            raise ValueError(
                f"optimization|objective_reference is {reference}; it must be positive, since it is "
                f"the magnitude the objective is scaled by."
            )
        return 1.0 / reference

    def _validate_objective(self) -> None:
        """Check that the objective names something the model produces."""
        available = sorted({*MISSION_OUTPUTS_BY_NAME, *self.SIZING_OBJECTIVES})
        if self._objective not in available:
            raise ValueError(
                f"'{self._objective}' is not a quantity this model produces, so it cannot be minimized. "
                f"Available objectives: {available}"
            )

    def _read_design_variables(self) -> tuple[DesignVariable, ...]:
        """Read the design variables from configuration.

        Each is declared as a branch holding ``lower`` and ``upper``, and optionally
        ``scaler``:

        .. code-block:: yaml

           optimization:
             design_variables:
               ac:
                 geom:
                   wing:
                     S_ref:
                       lower: {value: 90.0, units: m**2}
                       upper: {value: 180.0, units: m**2}

        Returns
        -------
        tuple of DesignVariable
            In sorted name order.

        Raises
        ------
        ValueError
            If none are configured, or if one is missing a bound. An optimization with no
            design variables runs, converges immediately, and reports the baseline as an
            optimum.
        """
        prefix = f"{self.DESIGN_VARIABLE_ROOT}|"
        declared: dict[str, dict[str, str]] = {}
        for path in self._config:
            if not path.startswith(prefix):
                continue
            variable_name, _, attribute = path[len(prefix) :].rpartition("|")
            declared.setdefault(variable_name, {})[attribute] = path

        if not declared:
            raise ValueError(
                f"No design variables are configured under '{self.DESIGN_VARIABLE_ROOT}'. An "
                f"optimization with nothing to vary converges immediately and reports the baseline "
                f"design as an optimum."
            )

        variables = []
        for name, attributes in sorted(declared.items()):
            missing = sorted({"lower", "upper"} - set(attributes))
            if missing:
                raise ValueError(
                    f"Design variable '{name}' is missing {missing}. An unbounded design variable "
                    f"lets the optimizer leave the range the model's methods are valid over."
                )
            units = self._config.units(attributes["lower"])
            variables.append(
                DesignVariable(
                    name=name,
                    lower=self._config.scalar(attributes["lower"]),
                    upper=self._config.scalar(attributes["upper"], units=units),
                    units=units,
                    scaler=self._config.scalar(attributes["scaler"]) if "scaler" in attributes else None,
                )
            )
        return tuple(variables)

    # -- running ------------------------------------------------------------------------

    def build(self) -> om.Problem:
        """Assemble the optimization problem.

        Returns
        -------
        openmdao.api.Problem
            Set up, with design variables, objective, certification constraints and driver
            registered.
        """
        problem = om.Problem()
        problem.driver = self._driver.build()

        model = problem.model
        self._loop._add_design_parameters(model)
        self._loop._add_aircraft_disciplines(model)
        self._loop._add_weight_closure(model)
        self._loop.blackbox.build(model)
        self._certification.build(model, self._loop.blackbox)
        self._certification.register(model, self._loop.blackbox)

        # Design variables and the objective are non-dimensionalized for the same reason the
        # constraints are: this problem's variables span a taper ratio of 0.25 and an engine
        # rating of 27,000 lbf, and an optimizer presented with that range spends its
        # progress on the large numbers.
        for variable in self._design_variables:
            model.add_design_var(
                variable.name,
                lower=variable.lower,
                upper=variable.upper,
                units=variable.units,
                scaler=variable.effective_scaler,
            )

        model.add_objective(self.objective_path(), scaler=self._objective_scaler())

        self._loop._configure_solvers(model)
        problem.setup(check=False)
        return problem

    def run(self, problem: om.Problem, verbose: bool = False) -> OptimizationResult:
        """Converge the baseline, then optimize, then report.

        The baseline is converged first so the optimizer starts from a feasible mission
        rather than from the configuration's starting guesses. Without it the first
        function evaluation is at a point the mission solver cannot converge, and the
        optimizer sees a failure it cannot distinguish from a bad design.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem returned by :meth:`build`.
        verbose : bool, optional
            Print the continuation schedule as it runs. Default ``False``.

        Returns
        -------
        OptimizationResult
            The outcome, the final design, and the certification traceability matrix.
        """
        self._loop.converge(problem, verbose=verbose)
        baseline_objective = float(problem.get_val(self.objective_path()).reshape(-1)[0])

        problem.run_driver()

        outcome = self._driver.outcome(problem.driver)
        design_variables = {
            variable.name: float(problem.get_val(variable.name, units=variable.units).reshape(-1)[0])
            for variable in self._design_variables
        }
        results = {
            name: value
            for name, value in self._loop.results(problem).items()
            if not hasattr(value, "size") or value.size == 1
        }

        return OptimizationResult(
            outcome=outcome,
            objective_name=self._objective,
            objective_value=float(problem.get_val(self.objective_path()).reshape(-1)[0]),
            baseline_objective=baseline_objective,
            design_variables=design_variables,
            results=results,
            traceability_matrix=self._certification.traceability_matrix(problem, self._loop.blackbox),
        )

    def __repr__(self) -> str:
        """Return a representation naming the objective, optimizer and variable count."""
        return (
            f"DesignProblem(objective={self._objective!r}, optimizer={self._driver.name!r}, "
            f"design_variables={len(self._design_variables)}, requirements={len(self._certification)})"
        )
