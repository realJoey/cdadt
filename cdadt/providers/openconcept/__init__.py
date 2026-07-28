"""Providers backed by the OpenConcept component library.

These are the only cdadt modules outside :mod:`cdadt.mission.blackbox` that import
OpenConcept, and they **wrap** it: they instantiate its components and promote their
variables. They never modify, subclass to patch, or monkey-patch anything in it.

Every method constant these providers need is read from the
:class:`~cdadt.core.configuration.AircraftConfiguration`, including constants for which
OpenConcept itself supplies an option default. A default sourced from a textbook is still a
number applied to a configuration that never mentioned it, so cdadt requires it explicitly
and records its provenance for the run report.
"""

from cdadt.providers.openconcept.aerodynamics import (
    JetTransportDragProvider,
    JetTransportMaximumLiftProvider,
)
from cdadt.providers.openconcept.geometry import TrapezoidalGeometryProvider
from cdadt.providers.openconcept.landing import LandingPerformanceProvider
from cdadt.providers.openconcept.propulsion import RubberizedTurbofanProvider
from cdadt.providers.openconcept.stability import TailVolumeCoefficientProvider
from cdadt.providers.openconcept.weights import (
    FuelBurnMassProvider,
    JetTransportEmptyWeightProvider,
)

__all__ = [
    "FuelBurnMassProvider",
    "JetTransportDragProvider",
    "JetTransportEmptyWeightProvider",
    "JetTransportMaximumLiftProvider",
    "LandingPerformanceProvider",
    "RubberizedTurbofanProvider",
    "TailVolumeCoefficientProvider",
    "TrapezoidalGeometryProvider",
]
