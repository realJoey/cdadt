"""The optimizer: what may change, what must hold, and what is being minimized.

All three come out of the case file. A variable becomes free by gaining an ``optimize:`` entry
where it is already declared, so a study that frees one more variable differs by three lines;
constraints and the objective are their own top-level blocks, mirroring the
``add_constraint``/``add_objective`` calls in OpenConcept's own optimizing examples.

Everything is checked before anything expensive starts. A cheap probe of the box is built first,
and every free variable is confirmed to be an independent variable the box accepts, the
objective and every constraint to be a quantity it publishes. A misspelled name is a message
naming it with suggestions, not an OpenMDAO error thrown out of ``setup`` after a minute of
building.

Converging before driving is not optional. The box is a Newton-solved implicit system that needs
a continuation ladder to reach a long-range mission from a cold start; the driver's first
function evaluation must begin from a converged aircraft, and every later one begins from its
predecessor.
"""

from __future__ import annotations

from typing import Any

import openmdao.api as om

from cdadt.analysis import SizingAnalysis
from cdadt.blackbox import OpenConceptSizingBox
from cdadt.certification import CertificationBasis, ConstraintResult
from cdadt.mission import MissionError
from cdadt.results import ResponseCatalog, SizingResults

__all__ = ["OptimizationError", "OptimizationOutcome", "Optimizer"]


class OptimizationError(Exception):
    """Raised when an optimization is not well posed, before any of it is run."""


class OptimizationOutcome:
    """The baseline, the optimum, and whether the constraints hold at it.

    Parameters
    ----------
    baseline : SizingResults
        The converged design before optimization.
    optimum : SizingResults
        The converged design after it.
    constraints : list of ConstraintResult
        Every constraint evaluated at the optimum.
    objective : str
        Name of the objective, as the case file gave it.
    sense : str
        ``"minimize"`` or ``"maximize"``.
    failed : bool
        Whether the driver reported failure.
    """

    def __init__(
        self,
        baseline: SizingResults,
        optimum: SizingResults,
        constraints: list[ConstraintResult],
        objective: str,
        sense: str,
        failed: bool,
    ) -> None:
        self._baseline = baseline
        self._optimum = optimum
        self._constraints = list(constraints)
        self._objective = objective
        self._sense = sense
        self._failed = bool(failed)

    @property
    def baseline(self) -> SizingResults:
        """The converged design before optimization."""
        return self._baseline

    @property
    def optimum(self) -> SizingResults:
        """The converged design after optimization."""
        return self._optimum

    @property
    def constraints(self) -> list[ConstraintResult]:
        """Every constraint, evaluated at the optimum."""
        return list(self._constraints)

    @property
    def objective(self) -> str:
        """Name of the objective."""
        return self._objective

    @property
    def sense(self) -> str:
        """``"minimize"`` or ``"maximize"``."""
        return self._sense

    @property
    def succeeded(self) -> bool:
        """Whether the driver converged and every constraint is satisfied.

        Both halves matter. A driver that reports success having stopped on its iteration limit
        inside an infeasible region has not solved the problem, and a report that calls that an
        optimum is wrong.
        """
        return not self._failed and all(result.satisfied for result in self._constraints)

    @property
    def violated(self) -> list[ConstraintResult]:
        """The constraints that are not satisfied at the optimum."""
        return [result for result in self._constraints if not result.satisfied]

    @property
    def active(self) -> list[ConstraintResult]:
        """The constraints the optimum sits on, and which therefore shaped the design."""
        return [result for result in self._constraints if result.active]

    def changes(self) -> dict[str, tuple[float, float, float]]:
        """Return ``name -> (baseline, optimum, percent change)`` for every scalar result."""
        baseline = self._baseline.scalars()
        optimum = self._optimum.scalars()
        return {
            name: (
                baseline[name],
                value,
                (value - baseline[name]) / baseline[name] * 100.0 if baseline[name] else float("nan"),
            )
            for name, value in optimum.items()
            if name in baseline
        }

    def __repr__(self) -> str:
        """Return a representation naming the objective and the outcome."""
        return f"OptimizationOutcome({self._objective!r}, succeeded={self.succeeded})"


class Optimizer:
    """Optimize an aircraft against the constraints its case file declares.

    Parameters
    ----------
    analysis : SizingAnalysis
        The sizing analysis to optimize. Its case file must declare an objective and at least
        one variable carrying an ``optimize:`` entry.

    Raises
    ------
    OptimizationError
        If the case declares no objective, or if a free variable, the objective or a constraint
        does not address something the black box publishes.
    """

    def __init__(self, analysis: SizingAnalysis) -> None:
        config = analysis.config
        if not config.is_optimization:
            raise OptimizationError(
                f"The case {config.path or 'given'} declares no 'objective', so there is nothing to "
                f"optimize. Use SizingAnalysis directly to size without optimizing."
            )
        self._analysis = analysis
        self._config = config
        self._catalog: ResponseCatalog = analysis.catalog
        # The basis belongs to the study, not to the act of optimizing: a sizing run is entitled
        # to ask whether the aeroplane it converged meets its certification basis, and did not
        # used to be able to without constructing an Optimizer to do it.
        self._basis = analysis.certification
        self._baseline_design: dict[str, float] = {}
        self._resolved: dict[str, str] = {}
        self._validate()

    # -- checking ------------------------------------------------------------------------

    def _validate(self) -> None:
        """Confirm everything the case declares addresses something the box publishes.

        Done against a cheaply built probe rather than the real model, so a misspelled name
        costs a fraction of a second instead of failing inside ``setup``.
        """
        probe = OpenConceptSizingBox.describe(self._config.black_box.model, options=self._config.black_box.options)
        conditions = self._analysis.performance.conditions

        # Resolved here, against a built probe, and cached: design variables are declared on the
        # group *before* setup, when the real box has no model to ask. A name that resolves
        # neither way is left as written so that ``check_settable`` reports it with suggestions,
        # which is more use than "neither X nor mission.X exists".
        self._resolved = {}
        for name in self._config.free_variables:
            try:
                self._resolved[name] = conditions.resolve(name, probe)
            except MissionError:
                self._resolved[name] = name
        probe.check_settable(list(self._resolved.values()))

        paths = [self.objective_path] + [constraint.path for constraint in self._basis]
        unknown = [path for path in paths if not probe.has(path)]
        if unknown:
            raise OptimizationError(
                "The black box does not publish these quantities, so they cannot be optimized or "
                f"constrained: {unknown}. Response names cdadt knows: {sorted(self._catalog)}."
            )

    # -- state ---------------------------------------------------------------------------

    @property
    def analysis(self) -> SizingAnalysis:
        """The sizing analysis being optimized."""
        return self._analysis

    @property
    def basis(self) -> CertificationBasis:
        """The constraints being enforced."""
        return self._basis

    @property
    def design_variables(self) -> dict[str, str]:
        """Return ``case-file name -> black-box path`` for every freed variable.

        The two differ whenever a mission value is freed: ``cruise|h0`` in a case file is
        ``mission.cruise|h0`` inside the box. Resolved once, against a cheap probe, because a
        design variable has to be declared before the real model exists to be asked.
        """
        return dict(self._resolved)

    @property
    def objective_path(self) -> str:
        """Return the black-box path of the objective."""
        return self._catalog.path(self._config.objective.name)

    @property
    def objective_units(self) -> str | None:
        """Return the units the objective is optimized in."""
        objective = self._config.objective
        return objective.units if objective.units is not None else self._catalog.units(objective.name)

    # -- running -------------------------------------------------------------------------

    def prepare(self, verbose: bool = False, driver: bool = True) -> SizingResults:
        """Declare the study on the box and converge the baseline design.

        Parameters
        ----------
        verbose : bool, optional
            Print the continuation steps and the driver's own progress. Default ``False``.
        driver : bool, optional
            Attach the driver the case file asks for. Passing ``False`` leaves the same
            declared, converged problem without one, which is what a total-derivative check
            needs: the derivatives an optimizer would step on, before it has stepped.

        Returns
        -------
        SizingResults
            The converged baseline, read before anything has been optimized.
        """
        self._analysis.build(register=[self._declare])
        if driver:
            self._analysis.box.problem.driver = self._driver(verbose=verbose)

        self._analysis.converge(verbose=verbose)
        baseline = self._analysis.results()

        # Read the free variables straight off the box: one need not be an airframe parameter at
        # all -- OpenConcept's own examples optimize mission-level values such as cruise
        # altitude -- and anything the box publishes as an independent variable is legal.
        box = self._analysis.box
        self._baseline_design = {
            name: float(box.get(self._resolved[name], units=spec.units))
            for name, spec in self._config.free_variables.items()
        }
        return baseline

    def run(self, verbose: bool = False) -> OptimizationOutcome:
        """Declare, converge the baseline, then drive.

        Parameters
        ----------
        verbose : bool, optional
            Print the continuation steps and the driver's own progress. Default ``False``.
        """
        baseline = self.prepare(verbose=verbose)
        box = self._analysis.box
        succeeded = box.run_driver()
        return OptimizationOutcome(
            baseline=baseline,
            optimum=self._analysis.results(),
            constraints=self._basis.evaluate(box),
            objective=self._config.objective.name,
            sense=self._config.objective.sense,
            failed=not succeeded,
        )

    def _declare(self, model: om.Group) -> None:
        """Declare the design variables, the constraints and the objective, before setup."""
        for name, spec in self._config.free_variables.items():
            model.add_design_var(self._resolved[name], **spec.optimize.as_kwargs(spec.units))

        self._basis.register(model)
        model.add_objective(self.objective_path, **self._config.objective.as_kwargs(self.objective_units))

    def _driver(self, verbose: bool) -> om.Driver:
        """Return the driver the case file asks for.

        SLSQP comes from SciPy and is always available. Anything else is resolved through
        pyOptSparse, which is optional; the error names what is missing rather than failing
        somewhere inside OpenMDAO.
        """
        settings = self._config.driver
        if settings.name.upper() == "SLSQP":
            driver: om.Driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=settings.tol, maxiter=settings.maxiter)
        else:
            try:
                driver = om.pyOptSparseDriver(optimizer=settings.name)
            except Exception as error:  # pragma: no cover - depends on the installed stack
                raise OptimizationError(
                    f"Optimizer '{settings.name}' requires pyOptSparse built with that optimizer. "
                    f"Install it, or set 'driver.name: SLSQP'."
                ) from error
            driver.opt_settings.update(self._pyoptsparse_settings())
        driver.options["debug_print"] = ["objs", "nl_cons"] if verbose else []
        return driver

    def _pyoptsparse_settings(self) -> dict[str, Any]:
        """Return the per-optimizer settings, with the case file's own on top."""
        settings = self._config.driver
        name = settings.name.upper()
        defaults: dict[str, Any] = {}
        if name == "IPOPT":
            defaults = {
                "max_iter": settings.maxiter,
                "tol": settings.tol,
                "print_level": 0,
                # The box's totals are dense and cheap next to the Newton solve, so a
                # limited-memory Hessian approximation is the right trade.
                "hessian_approximation": "limited-memory",
                "mu_strategy": "adaptive",
                # IPOPT scales the objective from its own gradient, which is why an objective in
                # kilograms needs no ref when this optimizer is used.
                "nlp_scaling_method": "gradient-based",
            }
        elif name == "SNOPT":
            defaults = {
                "Major iterations limit": settings.maxiter,
                "Major optimality tolerance": settings.tol,
            }
        defaults.update(settings.options)
        return defaults

    # -- reporting -----------------------------------------------------------------------

    def report(self, outcome: OptimizationOutcome) -> str:
        """Return the baseline-to-optimum comparison and the traceability matrix."""
        box = self._analysis.box
        settings = self._config.driver

        lines = [
            f"Objective : {outcome.sense} {outcome.objective}  ({self.objective_path})",
            f"Driver    : {settings.name}, {len(self._config.free_variables)} design variables, "
            f"{len(self._basis)} constraints",
            "",
            "Design variables",
            "----------------",
            f"{'variable':<32s} {'baseline':>14s} {'optimum':>14s} {'lower':>12s} {'upper':>12s}  units",
        ]
        for name, spec in self._config.free_variables.items():
            value = float(box.get(self._resolved[name], units=spec.units))
            start = self._baseline_design.get(name, float("nan"))
            lines.append(
                f"{name:<32s} {start:14.4f} {value:14.4f} "
                f"{float(spec.optimize.lower):12.4f} {float(spec.optimize.upper):12.4f}  {spec.units or '-'}"
            )

        lines += ["", "Results", "-------", f"{'quantity':<28s} {'baseline':>16s} {'optimum':>16s} {'change':>10s}"]
        for name, (start, value, percent) in outcome.changes().items():
            change = f"{percent:+9.2f}%" if percent == percent else "        -"
            lines.append(f"{name:<28s} {start:16.4f} {value:16.4f} {change:>10s}")

        lines += ["", "Constraints", "-----------", self._basis.traceability_matrix(box), ""]
        if outcome.succeeded:
            lines.append("Optimization SUCCEEDED: the driver converged and every constraint is met.")
        elif outcome.violated:
            lines.append(
                "Optimization DID NOT SUCCEED: violated "
                + ", ".join(result.constraint.name for result in outcome.violated)
            )
        else:
            lines.append("Optimization DID NOT SUCCEED: the driver reported failure.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation naming the objective and the counts."""
        return (
            f"Optimizer(objective={self._config.objective.name!r}, "
            f"design_variables={len(self._config.free_variables)}, constraints={len(self._basis)})"
        )
