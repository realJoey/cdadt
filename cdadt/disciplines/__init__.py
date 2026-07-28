"""Engineering disciplines, each a class that owns one domain's slice of the black box.

============================================================  =====================================
:class:`~cdadt.disciplines.geometry.Geometry`                 Wing, fuselage, nacelles, gear
:class:`~cdadt.disciplines.aerodynamics.Aerodynamics`         Span efficiency, airfoil, flaps, envelope
:class:`~cdadt.disciplines.propulsion.Propulsion`             Engine rating and count, installation weight
:class:`~cdadt.disciplines.stability.Stability`               Empennage shape and tail areas
:class:`~cdadt.disciplines.structures.Structures`             Load-carrying airframe mass breakdown
:class:`~cdadt.disciplines.weights.Weights`                   Payload, cabin, and the closed weight rollup
:class:`~cdadt.disciplines.performance.Performance`           The mission, and the field, climb and fuel results
============================================================  =====================================

Every one of them is a class with encapsulated state, reached only through properties and
methods that validate what they are given. None of them computes physics: the physics is inside
the black box. See :mod:`cdadt.disciplines.base` for what that division means and
:doc:`/architecture` for why it is drawn there.

:data:`AIRCRAFT_DISCIPLINES` is the set that describes the airframe and is composed by
:class:`~cdadt.aircraft.Aircraft`. Performance is not in it, because the mission is not a
property of the aircraft.
"""

from cdadt.disciplines.aerodynamics import Aerodynamics
from cdadt.disciplines.base import Discipline, DisciplineError
from cdadt.disciplines.geometry import Geometry
from cdadt.disciplines.performance import Performance
from cdadt.disciplines.propulsion import Propulsion
from cdadt.disciplines.stability import Stability
from cdadt.disciplines.structures import Structures
from cdadt.disciplines.weights import Weights

#: The disciplines that describe the airframe, in the order they are reported.
AIRCRAFT_DISCIPLINES: tuple[type[Discipline], ...] = (
    Geometry,
    Aerodynamics,
    Propulsion,
    Stability,
    Structures,
    Weights,
)

__all__ = [
    "AIRCRAFT_DISCIPLINES",
    "Aerodynamics",
    "Discipline",
    "DisciplineError",
    "Geometry",
    "Performance",
    "Propulsion",
    "Stability",
    "Structures",
    "Weights",
]
