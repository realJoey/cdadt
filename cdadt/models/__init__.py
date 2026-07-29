"""Physics cdadt owns, as distinct from physics it drives.

Everywhere else in cdadt, a class is an *interface* to the black box: a
:class:`~cdadt.disciplines.base.Discipline` declares which variables its domain sets and which
responses it reads, and computes nothing. This package is the exception, and the distinction is
deliberate.

A **model** here is something cdadt computes and *installs into* the analysis. The mission still
belongs to OpenConcept -- the trajectory, the balanced field, the reserves, the weight closure --
but the aerodynamic loads at every point of it can be cdadt's own, which is what makes this a
research framework rather than a driver.

Nothing in this package imports OpenConcept or openavl. It is OpenMDAO and numpy only, so the
models are testable, differentiable and readable without either dependency present. The classes
that bridge to a dependency live in :mod:`cdadt.adapter`, which is the only part of cdadt allowed
to import one.

Because these components are real physics, they carry analytic derivatives. An optimizer steps on
them and :doc:`/verification` publishes a derivative study; a finite-differenced partial here
would be a silent loss of accuracy in every gradient the framework reports.
"""

from cdadt.models.coefficients import AeroCoefficients
from cdadt.models.loads import AerodynamicLoads, FlightCondition, LoadsError, Planform
from cdadt.models.planform import TrapezoidalPlanform, WingSection
from cdadt.models.polar import PolarLoads

__all__ = [
    "AeroCoefficients",
    "AerodynamicLoads",
    "FlightCondition",
    "LoadsError",
    "Planform",
    "PolarLoads",
    "TrapezoidalPlanform",
    "WingSection",
]
