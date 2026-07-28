"""Optimizer drivers, behind an interface that can report whether the run succeeded.

The point of this abstraction is not to make optimizers interchangeable -- OpenMDAO already
does that. It is :meth:`OptimizerDriver.outcome`.

An OpenMDAO ``run_driver`` call returns whether it finished, not whether it succeeded. An
optimizer that hit its iteration limit, or stopped at a point it could not restore
feasibility from, returns a design vector and a set of results that look exactly like a
converged optimum. Reporting those as an optimum is the single easiest way for a design
study to be wrong, so every driver here is required to say what actually happened, and
:class:`~cdadt.optimization.problem.DesignProblem` refuses to call a non-optimal result an
optimum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import openmdao.api as om

from cdadt.core.configuration import AircraftConfiguration

__all__ = ["IpoptDriver", "OptimizationOutcome", "OptimizerDriver", "SlsqpDriver"]


@dataclass(frozen=True)
class OptimizationOutcome:
    """What an optimizer run actually achieved.

    Parameters
    ----------
    optimal : bool
        ``True`` only if the optimizer reports it converged to an optimum. Hitting an
        iteration limit, stalling, or failing to restore feasibility are all ``False``.
    status : str
        The optimizer's own description of how it exited.
    iterations : int or None
        Number of major iterations, if the optimizer reports it.
    """

    optimal: bool
    status: str
    iterations: int | None = None


class OptimizerDriver(ABC):
    """Base class for an optimizer cdadt can drive a design problem with.

    Parameters
    ----------
    config : AircraftConfiguration
        Configuration holding the optimizer's settings.
    """

    def __init__(self, config: AircraftConfiguration) -> None:
        self._config = config
        self.validate_configuration()

    @property
    def config(self) -> AircraftConfiguration:
        """Return the configuration this driver reads its settings from."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the optimizer's name, for reports."""

    @abstractmethod
    def build(self) -> om.Driver:
        """Return a configured OpenMDAO driver."""

    @abstractmethod
    def outcome(self, driver: om.Driver) -> OptimizationOutcome:
        """Report what the run achieved.

        Parameters
        ----------
        driver : openmdao.api.Driver
            The driver after ``run_driver`` has returned.

        Returns
        -------
        OptimizationOutcome
            Whether the result is an optimum, and how the optimizer exited.
        """

    def validate_configuration(self) -> None:  # noqa: B027 -- optional hook
        """Check that the configuration holds this driver's settings."""

    def __repr__(self) -> str:
        """Return a representation naming the optimizer."""
        return f"{type(self).__name__}(name={self.name!r})"


class IpoptDriver(OptimizerDriver):
    """IPOPT, through pyOptSparse.

    An interior-point method, which suits this problem: the certification constraints are
    nonlinear and several of them are active at the optimum, and interior-point methods
    handle that better than an active-set method does.

    Notes
    -----
    **Required configuration**, with no defaults: ``optimization|max_iterations``,
    ``optimization|tolerance``.

    IPOPT's ``optInform`` value is ``0`` for "solved" and ``1`` for "solved to acceptable
    level". cdadt treats only ``0`` as optimal. "Acceptable" means IPOPT relaxed its own
    tolerances because it stopped making progress, and a design study should say so rather
    than report the point as converged.
    """

    #: IPOPT exit code meaning it converged to the requested tolerance.
    SOLVED = 0

    @property
    def name(self) -> str:
        """Return ``"IPOPT"``."""
        return "IPOPT"

    def validate_configuration(self) -> None:
        """Require the iteration limit and tolerance."""
        self.config.require_all(["optimization|max_iterations", "optimization|tolerance"])

    def build(self) -> om.Driver:
        """Return a configured pyOptSparse driver running IPOPT."""
        driver = om.pyOptSparseDriver(optimizer="IPOPT")
        driver.options["print_results"] = False
        driver.opt_settings["max_iter"] = int(self.config.scalar("optimization|max_iterations"))
        driver.opt_settings["tol"] = self.config.scalar("optimization|tolerance")
        driver.opt_settings["print_level"] = 0
        # Without this IPOPT writes ipopt.out into the working directory on every run.
        driver.opt_settings["file_print_level"] = 0
        return driver

    def outcome(self, driver: om.Driver) -> OptimizationOutcome:
        """Report IPOPT's exit status.

        pyOptSparse reports it as a ``SolutionInform`` dataclass with ``value`` and
        ``message`` fields. Older releases used a plain dict, so both are read.
        """
        solution = getattr(driver, "pyopt_solution", None)
        inform = getattr(solution, "optInform", None) if solution is not None else None

        if inform is None:
            return OptimizationOutcome(
                optimal=False,
                status="IPOPT did not report an exit status; the run did not reach the optimizer.",
            )

        if isinstance(inform, dict):
            value, message = inform.get("value"), inform.get("text", "")
        else:
            value, message = getattr(inform, "value", None), getattr(inform, "message", "")

        return OptimizationOutcome(optimal=value == self.SOLVED, status=f"IPOPT exit {value}: {message}")


class SlsqpDriver(OptimizerDriver):
    """SLSQP, through SciPy.

    Bundled with SciPy, so it needs no additional build. Kept as the fallback that lets the
    repository run anywhere, not as an equal alternative: SLSQP is an active-set method and
    is less reliable on this problem than IPOPT.

    Notes
    -----
    **Required configuration**, with no defaults: ``optimization|max_iterations``,
    ``optimization|tolerance``.
    """

    @property
    def name(self) -> str:
        """Return ``"SLSQP"``."""
        return "SLSQP"

    def validate_configuration(self) -> None:
        """Require the iteration limit and tolerance."""
        self.config.require_all(["optimization|max_iterations", "optimization|tolerance"])

    def build(self) -> om.Driver:
        """Return a configured SciPy driver running SLSQP."""
        driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
        driver.options["maxiter"] = int(self.config.scalar("optimization|max_iterations"))
        driver.options["tol"] = self.config.scalar("optimization|tolerance")
        driver.options["disp"] = False
        return driver

    def outcome(self, driver: om.Driver) -> OptimizationOutcome:
        """Report SciPy's exit status."""
        # OpenMDAO wraps SciPy's result in a DriverResult that exists, with success=False,
        # before the driver has run. So "not successful" and "never ran" look identical
        # there, and the iteration count is what separates them. Reporting a never-run
        # driver as a failed optimization would be a smaller error than the reverse, but it
        # would still be wrong about what happened.
        if getattr(driver, "iter_count", 0) == 0:
            return OptimizationOutcome(
                optimal=False,
                status="SciPy did not report a result; the run did not reach the optimizer.",
            )

        result = getattr(driver, "result", None)
        return OptimizationOutcome(
            optimal=bool(getattr(result, "success", False)),
            status=f"SLSQP: {getattr(result, 'exit_status', 'no status reported')}",
            iterations=getattr(driver, "iter_count", None),
        )
