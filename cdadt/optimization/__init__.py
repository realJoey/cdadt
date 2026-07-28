"""Design optimization: the centralized problem definition and the drivers that run it.

:mod:`cdadt.optimization.problem`
    :class:`~cdadt.optimization.problem.DesignProblem`, the one place a cdadt optimization
    is assembled. It owns discipline assembly, the sizing loop, the certification basis,
    the design variables, the objective and the driver.

:mod:`cdadt.optimization.driver`
    :class:`~cdadt.optimization.driver.OptimizerDriver` and its implementations. The
    interface exists mainly for
    :meth:`~cdadt.optimization.driver.OptimizerDriver.outcome`: an OpenMDAO run reports
    that it finished, not that it succeeded, and a run that stopped at its iteration limit
    returns results indistinguishable from an optimum.
"""

from cdadt.optimization.driver import IpoptDriver, OptimizationOutcome, OptimizerDriver, SlsqpDriver
from cdadt.optimization.problem import DesignProblem, DesignVariable, OptimizationResult

__all__ = [
    "DesignProblem",
    "DesignVariable",
    "IpoptDriver",
    "OptimizationOutcome",
    "OptimizationResult",
    "OptimizerDriver",
    "SlsqpDriver",
]
