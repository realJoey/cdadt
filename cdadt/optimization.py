"""The optimizer: what may change, what must hold, and what is being minimized.

:class:`Optimizer` is the object that owns the coupling between the three. It is handed a
:class:`~cdadt.analysis.SizingAnalysis` -- which knows how to converge an aircraft against its
mission -- reads the design variables, the objective and the certification requirements out of
the case file, declares all three on the black box's group before setup, converges a baseline,
and drives.

Everything it declares is checked before anything expensive starts. A cheap probe of the black
box is built first, and every design variable is confirmed to be an independent variable the box
actually accepts, every constraint and the objective to be a quantity it actually publishes. A
misspelled design variable is a message naming it and suggesting the nearest match, not an
OpenMDAO error thrown out of ``setup`` after a minute of building.

Order of operations
-------------------

Converging before driving is not optional. The black box is a Newton-solved implicit system that
needs a continuation ladder to reach a long-range mission from a cold start. The driver's first
function evaluation must begin from a converged aircraft; every later one begins from its
predecessor, which is why the ladder is walked once rather than per iteration.
"""

from __future__ import annotations

from typing import Any

import openmdao.api as om

from cdadt.analysis import SizingAnalysis
from cdadt.blackbox import OpenConceptSizingBox
from cdadt.certification import CertificationBasis, RequirementResult
from cdadt.config import OptimizationConfig
from cdadt.results import ResponseCatalog, SizingResults

__all__ = ["OptimizationError", "OptimizationOutcome", "Optimizer"]


class OptimizationError(Exception):
    """Raised when an optimization is not well posed, before any of it is run."""


class OptimizationOutcome:
    """The baseline, the optimum, and whether the requirements hold at it.

    Parameters
    ----------
    baseline : SizingResults
        The converged design before optimization.
    optimum : SizingResults
        The converged design after it.
    requirements : sequence of RequirementResult
        Every requirement evaluated at the optimum.
    objective : str
        Name of the objective, as the case file gave it.
    sense : str
        ``"minimize"`` or ``"maximize"``.
    failed : bool
        Whether the driver reported failure, following OpenMDAO's convention.
    """

    def __init__(
        self,
        baseline: SizingResults,
        optimum: SizingResults,
        requirements: list[RequirementResult],
        objective: str,
        sense: str,
        failed: bool,
    ) -> None:
        self._baseline = baseline
        self._optimum = optimum
        self._requirements = list(requirements)
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
    def requirements(self) -> list[RequirementResult]:
        """Every requirement, evaluated at the optimum."""
        return list(self._requirements)

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
        """Whether the driver converged and every requirement is satisfied.

        Both halves matter. A driver that reports success having stopped on its iteration limit
        inside an infeasible region has not solved the problem, and a report that calls that an
        optimum is wrong.
        """
        return not self._failed and all(result.satisfied for result in self._requirements)

    @property
    def violated(self) -> list[RequirementResult]:
        """The requirements that are not satisfied at the optimum."""
        return [result for result in self._requirements if not result.satisfied]

    @property
    def active(self) -> list[RequirementResult]:
        """The requirements the optimum sits on, and which therefore shaped the design."""
        return [result for result in self._requirements if result.active]

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
    """Optimize an aircraft against a certification basis.

    Parameters
    ----------
    analysis : SizingAnalysis
        The sizing analysis to optimize. Its case file must declare an ``optimization`` section.

    Raises
    ------
    OptimizationError
        If the case declares no optimization, or if a design variable, the objective or a
        constraint does not address something the black box publishes.

    Examples
    --------
    >>> optimizer = Optimizer(SizingAnalysis(Config.from_yaml("cases/b738_optimization.yaml")))
    >>> outcome = optimizer.run()
    >>> print(optimizer.report(outcome))
    """

    def __init__(self, analysis: SizingAnalysis) -> None:
        settings = analysis.config.optimization
        if settings is None:
            raise OptimizationError(
                f"The case {analysis.config.path or 'given'} declares no 'optimization' section, so there "
                f"is nothing to optimize. Use SizingAnalysis directly to size without optimizing."
            )
        self._analysis = analysis
        self._settings: OptimizationConfig = settings
        self._catalog: ResponseCatalog = analysis.catalog
        self._basis = CertificationBasis.from_specs(settings.requirements)
        self._validate()

    # -- checking ------------------------------------------------------------------------

    def _validate(self) -> None:
        """Confirm everything the case declares addresses something the black box publishes.

        Done against a cheaply built probe of the box rather than the real model, so that a
        misspelled name costs a fraction of a second instead of failing inside ``setup``.
        """
        probe = OpenConceptSizingBox.describe(self._analysis.config.black_box.model)

        names = [variable.name for variable in self._settings.design_variables]
        probe.check_settable(names)

        responses = [self.objective_path] + [requirement.path(self._catalog) for requirement in self._basis]
        unknown = [path for path in responses if not probe.has(path)]
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
    def settings(self) -> OptimizationConfig:
        """The optimization section of the case file."""
        return self._settings

    @property
    def basis(self) -> CertificationBasis:
        """The certification basis being enforced."""
        return self._basis

    @property
    def objective_path(self) -> str:
        """Return the black-box path of the objective."""
        return self._catalog.path(self._settings.objective.name)

    @property
    def objective_units(self) -> str | None:
        """Return the units the objective is optimized in."""
        objective = self._settings.objective
        return objective.units if objective.units is not None else self._catalog.units(objective.name)

    # -- running -------------------------------------------------------------------------

    def run(self, verbose: bool = False) -> OptimizationOutcome:
        """Declare, converge the baseline, then drive.

        Parameters
        ----------
        verbose : bool, optional
            Print the continuation steps and the driver's own progress. Default ``False``.

        Returns
        -------
        OptimizationOutcome
            The baseline, the optimum, and the certification basis evaluated at it.
        """
        self._analysis.build(register=[self._declare])
        self._analysis.box.problem.driver = self._driver(verbose=verbose)

        self._analysis.converge(verbose=verbose)
        baseline = self._analysis.results()

        failed = self._analysis.box.run_driver()

        optimum = self._analysis.results()
        requirements = self._basis.evaluate(self._analysis.box, self._catalog)
        return OptimizationOutcome(
            baseline=baseline,
            optimum=optimum,
            requirements=requirements,
            objective=self._settings.objective.name,
            sense=self._settings.objective.sense,
            failed=bool(failed),
        )

    def _declare(self, model: om.Group) -> None:
        """Declare the design variables, the constraints and the objective, before setup."""
        for variable in self._settings.design_variables:
            model.add_design_var(
                variable.name,
                lower=variable.lower,
                upper=variable.upper,
                units=variable.units,
                ref=variable.ref,
            )

        self._basis.register(model, self._catalog)

        # OpenMDAO drivers always minimize. A maximization is therefore a minimization of the
        # objective divided by a negative reference, which is also where the case file's own
        # scaling is applied -- the two cannot both be given, so they are combined here.
        objective = self._settings.objective
        reference = (objective.ref if objective.ref is not None else 1.0) * objective.scaler
        model.add_objective(self.objective_path, units=self.objective_units, ref=reference)

    def _driver(self, verbose: bool) -> om.Driver:
        """Return the driver the case file asks for.

        SLSQP comes from SciPy and is always available. Anything else is resolved through
        pyOptSparse, which is optional; the error names what is missing rather than failing
        somewhere inside OpenMDAO.
        """
        name = self._settings.driver.upper()
        if name == "SLSQP":
            driver: om.Driver = om.ScipyOptimizeDriver(
                optimizer="SLSQP", tol=self._settings.tol, maxiter=self._settings.maxiter
            )
        else:
            try:
                driver = om.pyOptSparseDriver(optimizer=self._settings.driver)
            except Exception as error:  # pragma: no cover - depends on the installed stack
                raise OptimizationError(
                    f"Optimizer '{self._settings.driver}' requires pyOptSparse built with that optimizer. "
                    f"Install it, or set 'optimization.driver.name: SLSQP'."
                ) from error
            driver.opt_settings.update(self._pyoptsparse_settings())
        driver.options["debug_print"] = ["objs", "nl_cons"] if verbose else []
        return driver

    def _pyoptsparse_settings(self) -> dict[str, Any]:
        """Return the per-optimizer settings, with the case file's own on top."""
        name = self._settings.driver.upper()
        defaults: dict[str, Any] = {}
        if name == "IPOPT":
            defaults = {
                "max_iter": self._settings.maxiter,
                "tol": self._settings.tol,
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
                "Major iterations limit": self._settings.maxiter,
                "Major optimality tolerance": self._settings.tol,
            }
        defaults.update(self._settings.driver_options)
        return defaults

    # -- reporting -----------------------------------------------------------------------

    def report(self, outcome: OptimizationOutcome) -> str:
        """Return the baseline-to-optimum comparison and the traceability matrix."""
        lines = [
            f"Objective : {outcome.sense} {outcome.objective}  ({self.objective_path})",
            f"Driver    : {self._settings.driver}, {len(self._settings.design_variables)} design variables, "
            f"{len(self._basis)} requirements",
            "",
            "Design variables",
            "----------------",
            f"{'variable':<32s} {'baseline':>14s} {'optimum':>14s} {'lower':>12s} {'upper':>12s}  units",
        ]
        box = self._analysis.box
        for variable in self._settings.design_variables:
            value = float(box.get(variable.name, units=variable.units))
            start = self._analysis.aircraft.value(variable.name)
            lines.append(
                f"{variable.name:<32s} {start:14.4f} {value:14.4f} "
                f"{variable.lower:12.4f} {variable.upper:12.4f}  {variable.units or '-'}"
            )

        lines += ["", "Results", "-------", f"{'quantity':<28s} {'baseline':>16s} {'optimum':>16s} {'change':>10s}"]
        for name, (start, value, percent) in outcome.changes().items():
            change = f"{percent:+9.2f}%" if percent == percent else "        -"
            lines.append(f"{name:<28s} {start:16.4f} {value:16.4f} {change:>10s}")

        lines += ["", "Certification basis", "-------------------"]
        lines.append(self._basis.traceability_matrix(box, self._catalog))
        lines.append("")
        if outcome.succeeded:
            lines.append("Optimization SUCCEEDED: the driver converged and every requirement is met.")
        elif outcome.violated:
            lines.append(
                "Optimization DID NOT SUCCEED: violated "
                + ", ".join(result.requirement.name for result in outcome.violated)
            )
        else:
            lines.append("Optimization DID NOT SUCCEED: the driver reported failure.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation naming the objective and the counts."""
        return (
            f"Optimizer(objective={self._settings.objective.name!r}, "
            f"design_variables={len(self._settings.design_variables)}, requirements={len(self._basis)})"
        )
