"""cdadt -- a certification-driven aircraft design tool.

cdadt sizes and optimizes an aircraft against an explicit certification basis. The sizing
itself -- balanced-field takeoff, climb, cruise, descent, 14 CFR Part 25 reserves and loiter --
is performed by an OpenConcept analysis used as a **black box**: cdadt sets its inputs,
converges it, and reads its outputs. No OpenConcept source is modified, no OpenConcept class is
subclassed, and no cdadt module imports OpenConcept at all; the model is named in the case file
and loaded by name. See :doc:`/blackbox`.

Everything cdadt contributes is a class: the disciplines that own the interface, the mission
that is flown, the certification basis that must hold, and the optimizer that drives the box.
"""

from cdadt.blackbox import BlackBoxError, OpenConceptSizingBox, SolverSettings, VariableInfo
from cdadt.disciplines import (
    AIRCRAFT_DISCIPLINES,
    Aerodynamics,
    Discipline,
    DisciplineError,
    Geometry,
    Performance,
    Propulsion,
    Stability,
    Structures,
    Weights,
)
from cdadt.mission import ContinuationStep, MissionProfile, PhaseSchedule
from cdadt.parameters import Parameter, Response

__version__ = "0.2.0"

__all__ = [
    "AIRCRAFT_DISCIPLINES",
    "Aerodynamics",
    "BlackBoxError",
    "ContinuationStep",
    "Discipline",
    "DisciplineError",
    "Geometry",
    "MissionProfile",
    "OpenConceptSizingBox",
    "Parameter",
    "Performance",
    "PhaseSchedule",
    "Propulsion",
    "Response",
    "SolverSettings",
    "Stability",
    "Structures",
    "VariableInfo",
    "Weights",
    "__version__",
]
