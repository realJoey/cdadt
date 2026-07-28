"""The mission black box and the contract that bounds it.

:mod:`cdadt.mission.contract`
    Every OpenConcept variable name, path, and phase, declared once as data. Verified
    against a really-built OpenConcept model by the contract tests.

:mod:`cdadt.mission.aircraft_model`
    :class:`~cdadt.mission.aircraft_model.AircraftModelFactory`, which produces the class
    OpenConcept instantiates inside each phase, with a discipline set bound to it and no
    shared global state.

:mod:`cdadt.mission.blackbox`
    :class:`~cdadt.mission.blackbox.MissionBlackBox`, the only class in cdadt that imports
    from ``openconcept.mission``.
"""

from cdadt.mission.aircraft_model import AircraftModelFactory, CdadtAircraftModel
from cdadt.mission.blackbox import ContinuationStep, MissionBlackBox, MissionProfile, PhaseSchedule
from cdadt.mission.contract import (
    AIRCRAFT_MODEL_OUTPUTS,
    MISSION_OUTPUTS,
    MISSION_PHASES,
    MISSION_PROFILE_INPUTS,
    MissionOutput,
    PhaseKind,
    PhaseSpec,
    phase_supplied_inputs,
)

__all__ = [
    "AIRCRAFT_MODEL_OUTPUTS",
    "MISSION_OUTPUTS",
    "MISSION_PHASES",
    "MISSION_PROFILE_INPUTS",
    "AircraftModelFactory",
    "CdadtAircraftModel",
    "ContinuationStep",
    "MissionBlackBox",
    "MissionOutput",
    "MissionProfile",
    "PhaseKind",
    "PhaseSchedule",
    "PhaseSpec",
    "phase_supplied_inputs",
]
