"""The engineering disciplines cdadt models.

Each class here is one domain, with its own state, delegating its physics to a
:class:`~cdadt.core.provider.Provider`. The classes are deliberately thin: a discipline's
job is to declare its identity and scope and to hold its provider, not to compute. That is
what lets an empirical buildup and a high-fidelity analysis be interchanged without the
surrounding model knowing.

Disciplines are split by :class:`~cdadt.core.discipline.DisciplineScope`:

Aircraft-scoped -- built once, above the mission
    :class:`~cdadt.disciplines.geometry.Geometry`,
    :class:`~cdadt.disciplines.stability.Stability`,
    :class:`~cdadt.disciplines.weights.EmptyWeight`, and
    :class:`~cdadt.disciplines.aerodynamics.MaximumLift`. These describe the airframe, not
    a flight condition.

Phase-scoped -- built inside every mission phase
    :class:`~cdadt.disciplines.aerodynamics.Aerodynamics`,
    :class:`~cdadt.disciplines.propulsion.Propulsion`, and
    :class:`~cdadt.disciplines.weights.MassBookkeeping`. These depend on the flight
    condition and are evaluated at every analysis node.
"""

from cdadt.disciplines.aerodynamics import Aerodynamics, MaximumLift
from cdadt.disciplines.geometry import Geometry
from cdadt.disciplines.landing import LandingPerformance
from cdadt.disciplines.propulsion import Propulsion
from cdadt.disciplines.stability import Stability
from cdadt.disciplines.weights import EmptyWeight, MassBookkeeping

__all__ = [
    "Aerodynamics",
    "EmptyWeight",
    "Geometry",
    "LandingPerformance",
    "MassBookkeeping",
    "MaximumLift",
    "Propulsion",
    "Stability",
]
