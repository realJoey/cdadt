"""Engineering disciplines, each a class that owns its own analysis.

Aircraft-scoped disciplines describe the airframe and are evaluated once:

===================================================  ========================================
:class:`~cdadt.disciplines.geometry.Geometry`        MAC, tail lever arms, wetted areas
:class:`~cdadt.disciplines.stability.Stability`      Empennage areas by tail volume coefficient
:class:`~cdadt.disciplines.high_lift.HighLift`       Clean and takeoff maximum lift coefficients
:class:`~cdadt.disciplines.weights.Weights`          Operating empty weight, landing weight
===================================================  ========================================

Phase-scoped disciplines depend on the flight condition and are evaluated at every node of
every mission phase:

=========================================================  ==================================
:class:`~cdadt.disciplines.aerodynamics.Aerodynamics`      Parasite drag buildup, drag polar
:class:`~cdadt.disciplines.propulsion.Propulsion`          Engine deck, installed thrust and fuel flow
:class:`~cdadt.disciplines.mass.MassProperties`            Fuel burn integration, instantaneous weight
=========================================================  ==================================
"""

from cdadt.disciplines.aerodynamics import Aerodynamics
from cdadt.disciplines.base import AircraftDiscipline, Discipline, PhaseDiscipline
from cdadt.disciplines.geometry import Geometry
from cdadt.disciplines.high_lift import HighLift
from cdadt.disciplines.mass import MassProperties
from cdadt.disciplines.propulsion import Propulsion
from cdadt.disciplines.stability import Stability
from cdadt.disciplines.weights import Weights

__all__ = [
    "Aerodynamics",
    "AircraftDiscipline",
    "Discipline",
    "Geometry",
    "HighLift",
    "MassProperties",
    "PhaseDiscipline",
    "Propulsion",
    "Stability",
    "Weights",
]
