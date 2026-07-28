"""The design optimizer: design variables, an objective, and a certification basis.

The optimizer is the object that owns the coupling between what may change and what must hold.
It is handed a :class:`~cdadt.sizing.SizingAnalysis` -- which knows how to converge the
aircraft against its mission -- a set of :class:`DesignVariable`\\ s, an objective, and a
:class:`~cdadt.certification.CertificationBasis`. It registers all three with the model before
setup, converges the baseline, and drives.

Converging before driving is not optional. The mission is a Newton-solved implicit system that
needs a continuation schedule to reach the design profile from a cold start; the optimizer's
first function evaluation must begin from a converged aircraft, and every later one begins
from its predecessor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import openmdao.api as om

from cdadt.certification import CertificationBasis
from cdadt.sizing import SizingAnalysis

__all__ = ["DesignOptimizer", "DesignVariable"]


@dataclass(frozen=True)
class DesignVariable:
    """One quantity the optimizer may change.

    Parameters
    ----------
    name : str
        Problem path of the variable, e.g. ``"ac|geom|wing|S_ref"``. It must be an
        independent variable -- something the aircraft definition publishes -- because
        anything the model computes is not free to choose.
    lower, upper : float
        Bounds, in ``units``.
    units : str or None, optional
        Units of the bounds. ``None`` for a dimensionless variable.
    ref : float or None, optional
        Value that scales the variable to order one for the optimizer. Defaults to ``upper``,
        which is almost always the right order of magnitude and is the difference between an
        optimizer that converges and one that stalls when wing area in square metres and
        aspect ratio share a design space.
    """

    name: str
    lower: float
    upper: float
    units: str | None = None
    ref: float | None = None

    @property
    def scale(self) -> float:
        """Return the reference value used to scale this variable."""
        return float(self.ref if self.ref is not None else self.upper)


class DesignOptimizer:
    """Minimize an objective over an aircraft subject to a certification basis.

    Parameters
    ----------
    analysis : SizingAnalysis
        The sizing analysis to optimize. Its aircraft, mission and solver settings are used
        as given.
    design_variables : sequence of DesignVariable
        What the optimizer may change.
    objective : str
        What to minimize. Either a key of :attr:`~cdadt.sizing.SizingAnalysis.RESULTS` --
        ``"total_fuel"``, ``"MTOW"`` -- or a raw problem path.
    certification : CertificationBasis, optional
        Requirements to enforce. An optimization with no basis is a valid thing to run, and
        the report says so rather than printing an empty table.
    optimizer : str, optional
        ``"SLSQP"`` (SciPy, always available) or a pyOptSparse optimizer name such as
        ``"IPOPT"`` or ``"SNOPT"``. Default ``"SLSQP"``.
    maxiter : int, optional
        Optimizer iteration limit. Default 50.
    tol : float, optional
        Optimizer convergence tolerance. Default 1e-6.
    objective_ref : float or None, optional
        Value scaling the objective to order one, e.g. ``2e4`` for a fuel mass in kilograms.
        Default ``None``, meaning no scaling. It matters: SLSQP takes its finite-difference
        step and its convergence test on the *scaled* objective, so an objective of order
        :math:`10^4` is effectively converged before it starts. IPOPT scales the objective
        from its own gradient and does not need this.

    Examples
    --------
    >>> optimizer = DesignOptimizer(analysis, design_variables, "total_fuel", basis)
    >>> result = optimizer.run()
    >>> print(optimizer.report())
    """

    def __init__(
        self,
        analysis: SizingAnalysis,
        design_variables: Sequence[DesignVariable],
        objective: str,
        certification: CertificationBasis | None = None,
        optimizer: str = "SLSQP",
        maxiter: int = 50,
        tol: float = 1e-6,
        objective_ref: float | None = None,
    ) -> None:
        self.analysis = analysis
        self.design_variables = tuple(design_variables)
        self.objective = objective
        self.certification = certification
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.tol = tol
        self.objective_ref = objective_ref
        self._baseline: dict[str, float] | None = None
        if not self.design_variables:
            raise ValueError("An optimization needs at least one design variable.")

    # -- inspection ---------------------------------------------------------------------

    @property
    def objective_path(self) -> str:
        """Return the problem path of the objective.

        A key of :attr:`~cdadt.sizing.SizingAnalysis.RESULTS` resolves to its path; anything
        else is taken as a path already.
        """
        if self.objective in SizingAnalysis.RESULTS:
            return SizingAnalysis.RESULTS[self.objective][0]
        return self.objective

    @property
    def objective_units(self) -> str | None:
        """Return the units the objective is minimized in."""
        if self.objective in SizingAnalysis.RESULTS:
            return SizingAnalysis.RESULTS[self.objective][1]
        return None

    @property
    def baseline(self) -> dict[str, float]:
        """Return the converged results before optimization.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called.
        """
        if self._baseline is None:
            raise RuntimeError("The baseline has not been evaluated yet; call run().")
        return self._baseline

    # -- running ------------------------------------------------------------------------

    def run(self, verbose: bool = False) -> dict[str, float]:
        """Build, converge the baseline, then optimize.

        Parameters
        ----------
        verbose : bool, optional
            Print the continuation steps and the optimizer's own output. Default ``False``.

        Returns
        -------
        dict
            The optimized results, in the same form as
            :meth:`~cdadt.sizing.SizingAnalysis.results`.
        """
        problem = self.analysis.build(hooks=[self._register])
        self._attach_driver(problem, verbose=verbose)

        self.analysis.converge(verbose=verbose)
        self._baseline = self.analysis.results()

        problem.run_driver()
        return self.analysis.results()

    def _register(self, model: om.Group) -> None:
        """Register design variables, constraints and the objective before setup."""
        for variable in self.design_variables:
            model.add_design_var(
                variable.name,
                lower=variable.lower,
                upper=variable.upper,
                units=variable.units,
                ref=variable.scale,
            )

        if self.certification is not None:
            self.certification.build(model)
            self.certification.register(model)

        model.add_objective(
            self.objective_path,
            units=self.objective_units,
            ref=self.objective_ref if self.objective_ref is not None else 1.0,
        )

    def _attach_driver(self, problem: om.Problem, verbose: bool) -> None:
        """Attach the requested optimizer.

        SLSQP comes from SciPy and is always available. Anything else is resolved through
        pyOptSparse, which is optional; the error names what is missing rather than failing
        inside OpenMDAO.
        """
        if self.optimizer.upper() == "SLSQP":
            driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=self.tol, maxiter=self.maxiter)
        else:
            try:
                driver = om.pyOptSparseDriver(optimizer=self.optimizer)
            except Exception as error:  # pragma: no cover - depends on the installed stack
                raise RuntimeError(
                    f"Optimizer '{self.optimizer}' requires pyOptSparse with that optimizer built in. "
                    f"Install it, or use optimizer='SLSQP'."
                ) from error
            driver.opt_settings.update(self._pyoptsparse_settings())
        driver.options["debug_print"] = ["objs", "nl_cons"] if verbose else []
        problem.driver = driver

    def _pyoptsparse_settings(self) -> dict[str, object]:
        """Return per-optimizer settings for pyOptSparse."""
        if self.optimizer.upper() == "IPOPT":
            return {
                "max_iter": self.maxiter,
                "tol": self.tol,
                "print_level": 0,
                # The sizing model's totals are dense and cheap relative to the Newton solve,
                # so a limited-memory Hessian approximation is the right trade.
                "hessian_approximation": "limited-memory",
                "mu_strategy": "adaptive",
                # IPOPT scales the objective from its own gradient, which is why an objective
                # in kilograms needs no ref when this optimizer is used.
                "nlp_scaling_method": "gradient-based",
            }
        if self.optimizer.upper() == "SNOPT":
            return {"Major iterations limit": self.maxiter, "Major optimality tolerance": self.tol}
        return {}

    # -- reporting ----------------------------------------------------------------------

    def report(self) -> str:
        """Return a baseline-versus-optimum table and the traceability matrix."""
        optimum = self.analysis.results()
        baseline = self.baseline

        lines = [
            f"{'quantity':<24s} {'baseline':>16s} {'optimum':>16s} {'change':>10s}",
            "-" * 70,
        ]
        for name, value in optimum.items():
            start = baseline[name]
            change = f"{(value - start) / start * 100:+9.2f}%" if start != 0.0 else "        -"
            lines.append(f"{name:<24s} {start:16.4f} {value:16.4f} {change:>10s}")

        lines += ["", f"Objective: minimize {self.objective} ({self.objective_path})", ""]
        if self.certification is None:
            lines.append("No certification basis was attached; nothing was constrained.")
        else:
            lines.append(self.certification.traceability_matrix(self.analysis.problem))
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation naming the objective and the variable count."""
        return (
            f"DesignOptimizer(objective={self.objective!r}, "
            f"design_variables={len(self.design_variables)}, optimizer={self.optimizer!r})"
        )
