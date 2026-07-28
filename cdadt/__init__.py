"""cdadt -- a certification-driven aircraft design tool.

cdadt sizes and optimizes an aircraft against an explicit certification basis. The mission
analysis it sizes against -- balanced-field takeoff, climb, cruise, descent, Part 25 reserves
and loiter -- comes from `OpenConcept <https://github.com/mdolab/openconcept>`_ and is used as
a black box: cdadt sets its inputs, converges it, and reads its outputs. No OpenConcept source
is modified, subclassed to change behaviour, or reimplemented.

The four objects most users need:

:class:`~cdadt.aircraft.AircraftDefinition`
    The design parameters (``ac|...``) as encapsulated state, loadable from YAML.
:class:`~cdadt.mission.MissionProfile`
    The mission to fly and the continuation schedule that converges it.
:class:`~cdadt.sizing.SizingAnalysis`
    Builds, converges and reports the coupled sizing problem.
:class:`~cdadt.optimization.DesignOptimizer`
    Adds design variables, an objective and certification constraints, and drives it.
"""

from cdadt.aircraft import AircraftDefinition
from cdadt.certification import (
    ApproachSpeed,
    BalancedFieldLength,
    CertificationBasis,
    EngineOutClimbGradient,
    Requirement,
    RequirementResult,
    Sense,
    ThrottleMargin,
)
from cdadt.mission import MissionProfile, PhaseSchedule
from cdadt.model import JetTransportPhaseModel, Part25PhaseModel, SizingModel
from cdadt.optimization import DesignOptimizer, DesignVariable
from cdadt.sizing import SizingAnalysis

__version__ = "0.2.0"

__all__ = [
    "AircraftDefinition",
    "ApproachSpeed",
    "BalancedFieldLength",
    "CertificationBasis",
    "DesignOptimizer",
    "DesignVariable",
    "EngineOutClimbGradient",
    "JetTransportPhaseModel",
    "MissionProfile",
    "Part25PhaseModel",
    "PhaseSchedule",
    "Requirement",
    "RequirementResult",
    "Sense",
    "SizingAnalysis",
    "SizingModel",
    "ThrottleMargin",
    "__version__",
]
