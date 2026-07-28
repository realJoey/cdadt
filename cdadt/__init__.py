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

from cdadt.aircraft import Aircraft, AircraftError
from cdadt.analysis import SizingAnalysis
from cdadt.blackbox import BlackBoxError, OpenConceptSizingBox, SolverSettings, VariableInfo
from cdadt.certification import (
    SHIPPED_REQUIREMENTS,
    BalancedFieldLength,
    CertificationBasis,
    DesignRange,
    EngineOutClimbGradient,
    MaximumTakeoffWeight,
    Requirement,
    RequirementCatalog,
    RequirementError,
    RequirementResult,
    ResponseLimit,
    ThrottleLimit,
)
from cdadt.config import (
    BlackBoxConfig,
    Config,
    ConfigError,
    DesignVariableSpec,
    MissionConfig,
    ObjectiveSpec,
    OptimizationConfig,
    RequirementSpec,
    SolverConfig,
)
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
from cdadt.optimization import OptimizationError, OptimizationOutcome, Optimizer
from cdadt.parameters import Parameter, Response
from cdadt.results import ResponseCatalog, SizingResults

__version__ = "0.2.0"

__all__ = [
    "AIRCRAFT_DISCIPLINES",
    "SHIPPED_REQUIREMENTS",
    "Aerodynamics",
    "Aircraft",
    "AircraftError",
    "BalancedFieldLength",
    "BlackBoxConfig",
    "BlackBoxError",
    "CertificationBasis",
    "Config",
    "ConfigError",
    "ContinuationStep",
    "DesignRange",
    "DesignVariableSpec",
    "Discipline",
    "DisciplineError",
    "EngineOutClimbGradient",
    "Geometry",
    "MaximumTakeoffWeight",
    "MissionConfig",
    "MissionProfile",
    "ObjectiveSpec",
    "OpenConceptSizingBox",
    "OptimizationConfig",
    "OptimizationError",
    "OptimizationOutcome",
    "Optimizer",
    "Parameter",
    "Performance",
    "PhaseSchedule",
    "Propulsion",
    "Requirement",
    "RequirementCatalog",
    "RequirementError",
    "RequirementResult",
    "RequirementSpec",
    "Response",
    "ResponseCatalog",
    "ResponseLimit",
    "SizingAnalysis",
    "SizingResults",
    "SolverConfig",
    "SolverSettings",
    "Stability",
    "Structures",
    "ThrottleLimit",
    "VariableInfo",
    "Weights",
    "__version__",
]
