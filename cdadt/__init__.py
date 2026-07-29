"""cdadt -- a certification-driven aircraft design tool.

cdadt sizes and optimizes an aircraft against an explicit set of constraints. The sizing
itself -- balanced-field takeoff, climb, cruise, descent, 14 CFR Part 25 reserves and loiter --
is performed by an OpenConcept analysis used as a **black box**: cdadt sets its inputs,
converges it, and reads its outputs. No OpenConcept source is modified, no OpenConcept class is
subclassed, and no cdadt module imports OpenConcept at all; the model is named in the case file
and loaded by name. See :doc:`/blackbox`.

A study is one YAML file, laid out like OpenConcept's own B738 run scripts: ``design_variables``
is the block ``B738.py`` builds with ``dv_comp.add_output_from_dict(...)``, ``initial_conditions``
is its ``set_values(prob, num_nodes)``, and ``continuation`` is the ladder inside
``B738_sizing.py``'s ``set_mission_profile``. Everything cdadt contributes is a class.
"""

from cdadt.aircraft import Aircraft, AircraftError
from cdadt.analysis import SizingAnalysis
from cdadt.artifacts import ArtifactError, MissionTrajectory, StudyArtifacts, Trace
from cdadt.blackbox import BlackBoxError, OpenConceptSizingBox, SolverSettings, VariableInfo
from cdadt.certification import CertificationBasis, Constraint, ConstraintError, ConstraintResult
from cdadt.config import (
    BlackBoxConfig,
    Bounds,
    Config,
    ConfigError,
    ConstraintSpec,
    DriverConfig,
    ObjectiveSpec,
    OptimizeSpec,
    Scaling,
    SolverConfig,
    VariableSpec,
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
from cdadt.mission import (
    ContinuationLadder,
    ContinuationStep,
    InitialConditions,
    MissionError,
)
from cdadt.optimization import OptimizationError, OptimizationOutcome, Optimizer
from cdadt.parameters import Parameter, Response
from cdadt.results import ResponseCatalog, SizingResults

__version__ = "0.3.0"

__all__ = [
    "AIRCRAFT_DISCIPLINES",
    "Aerodynamics",
    "Aircraft",
    "AircraftError",
    "ArtifactError",
    "BlackBoxConfig",
    "BlackBoxError",
    "Bounds",
    "CertificationBasis",
    "Config",
    "ConfigError",
    "Constraint",
    "ConstraintError",
    "ConstraintResult",
    "ConstraintSpec",
    "ContinuationLadder",
    "ContinuationStep",
    "Discipline",
    "DisciplineError",
    "DriverConfig",
    "Geometry",
    "InitialConditions",
    "MissionError",
    "MissionTrajectory",
    "ObjectiveSpec",
    "OpenConceptSizingBox",
    "OptimizationError",
    "OptimizationOutcome",
    "OptimizeSpec",
    "Optimizer",
    "Parameter",
    "Performance",
    "Propulsion",
    "Response",
    "ResponseCatalog",
    "Scaling",
    "SizingAnalysis",
    "SizingResults",
    "SolverConfig",
    "SolverSettings",
    "Stability",
    "Structures",
    "StudyArtifacts",
    "Trace",
    "VariableInfo",
    "VariableSpec",
    "Weights",
    "__version__",
]
