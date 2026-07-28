"""The cdadt/OpenConcept interface contract, declared once as data.

OpenConcept's mission analysis is a black box in the sense that cdadt does not reimplement
any of its physics and never modifies its source. It is **not** a black box in the sense of
a pure function: :class:`openconcept.mission.FullMissionWithReserve` takes an
``aircraft_model`` *class* as an option and instantiates it inside every phase --

.. code-block:: python

   # openconcept/mission/phases.py, five identical call sites
   self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"])

-- so OpenConcept calls back into cdadt once per phase. The boundary is therefore a
*contract*: OpenConcept supplies flight conditions and consumes forces and mass, and cdadt
supplies a group that turns one into the other.

This module states that contract as data, in one place. Nothing else in cdadt hardcodes an
OpenConcept variable name or path. The consequences:

* :mod:`cdadt.tests.mission.test_contract` verifies **every** declaration here against a
  really-built ``FullMissionWithReserve``. When OpenConcept changes, that test fails with
  the name that moved, rather than the model silently losing a connection.
* Replacing the mission black box means rewriting this module and
  :mod:`cdadt.mission.blackbox`, and nothing else.

References
----------
Variable meanings are documented by OpenConcept itself in the class docstrings of
``openconcept.mission.phases`` (``GroundRollPhase``, ``RobustRotationPhase``,
``SteadyFlightPhase``, ``ClimbAnglePhase``) and in ``doc/features/mission.rst``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cdadt.core.variables import Variable, VariableSet

__all__ = [
    "AIRCRAFT_MODEL_OUTPUTS",
    "MISSION_OUTPUTS",
    "MISSION_PHASES",
    "MISSION_PROFILE_INPUTS",
    "MissionOutput",
    "PhaseKind",
    "PhaseSpec",
    "phase_supplied_inputs",
]


# ==============================================================================
# What the aircraft model must produce
# ==============================================================================
THRUST = Variable(
    "thrust",
    "N",
    vectorized=True,
    description="Total thrust from all propulsors, along the flight path",
)
DRAG = Variable(
    "drag",
    "N",
    vectorized=True,
    description="Total drag in the airplane axis from all sources",
)
WEIGHT = Variable(
    "weight",
    "kg",
    vectorized=True,
    description="Instantaneous aircraft mass; OpenConcept calls this 'weight' but the units are mass",
)

AIRCRAFT_MODEL_OUTPUTS = VariableSet([THRUST, DRAG, WEIGHT])
"""Variables the cdadt aircraft model must promote as outputs in **every** phase.

OpenConcept consumes all three in the acceleration and steady-flight balances. A model that
fails to produce one of them does not fail loudly: the name simply never resolves and the
consuming component holds an unconnected input at its default. The contract test asserts
each of these exists as a promoted output in every phase.
"""


# ==============================================================================
# What OpenConcept supplies to the aircraft model
# ==============================================================================
# Flight conditions produced by ComputeAtmosphericProperties and Groundspeeds, present in
# every phase. Units are those OpenConcept declares.
_ATMOSPHERIC_CONDITIONS = [
    Variable("fltcond|h", "m", vectorized=True, description="Altitude"),
    Variable("fltcond|T", "K", vectorized=True, description="Static temperature"),
    Variable("fltcond|p", "Pa", vectorized=True, description="Static pressure"),
    Variable("fltcond|rho", "kg * m**-3", vectorized=True, description="Density"),
    Variable("fltcond|a", "m/s", vectorized=True, description="Speed of sound"),
    Variable("fltcond|q", "N * m**-2", vectorized=True, description="Dynamic pressure"),
    Variable("fltcond|M", None, vectorized=True, description="Mach number"),
    Variable("fltcond|Utrue", "m/s", vectorized=True, description="True airspeed"),
    Variable("fltcond|Ueas", "m/s", vectorized=True, description="Equivalent airspeed"),
]

_KINEMATIC_CONDITIONS = [
    Variable("fltcond|vs", "m/s", vectorized=True, description="Vertical speed"),
    Variable("fltcond|groundspeed", "m/s", vectorized=True, description="Ground speed"),
    Variable("fltcond|cosgamma", None, vectorized=True, description="Cosine of the flight path angle"),
    Variable("fltcond|singamma", None, vectorized=True, description="Sine of the flight path angle"),
]

_CONTROL_INPUTS = [
    Variable("fltcond|CL", None, vectorized=True, description="Lift coefficient"),
    Variable("throttle", None, vectorized=True, description="Throttle setting, 0 to slightly above 1"),
    Variable(
        "propulsor_active",
        None,
        vectorized=True,
        description="1.0 when all propulsors operate, 0.0 when one is failed",
    ),
]

_GROUND_ROLL_ONLY = [
    Variable(
        "braking",
        None,
        vectorized=True,
        description="Rolling or braking friction coefficient; nonphysical if applied in the air",
    ),
]

_TIME = Variable("duration", "s", description="Elapsed time of the phase")


class PhaseKind(Enum):
    """The OpenConcept phase class a mission phase is an instance of.

    The aircraft model receives a different set of supplied variables depending on which
    class instantiated it, which is why :meth:`~cdadt.core.provider.Provider.build`
    receives the phase name.
    """

    GROUND_ROLL = "GroundRollPhase"
    ROTATION = "RobustRotationPhase"
    STEADY_FLIGHT = "SteadyFlightPhase"
    CLIMB_ANGLE = "ClimbAnglePhase"


@dataclass(frozen=True)
class PhaseSpec:
    """One mission phase, as OpenConcept builds it.

    Parameters
    ----------
    name : str
        The string OpenConcept passes to the aircraft model as ``flight_phase``. Providers
        branch on this to decide the aircraft's configuration.
    subsystem : str
        The subsystem name under the mission group, used to build variable paths. This is
        the same as ``name`` for every phase except the engine-out climb-angle check, which
        OpenConcept adds as ``engineoutclimb`` while passing ``flight_phase`` as
        ``EngineOutClimbAngle`` (``openconcept/mission/profiles.py`` lines 103-107).
    kind : PhaseKind
        Which OpenConcept phase class instantiates the aircraft model here.
    in_takeoff : bool
        ``True`` for the balanced-field takeoff phases, where the aircraft is in takeoff
        configuration (flaps deployed, gear down) and on or near the ground.
    is_reserve : bool
        ``True`` for the Part 25 reserve mission and loiter phases.
    integrates_fuel : bool
        ``True`` if fuel burn accumulates over the phase and carries into the next one.
        The takeoff phases and the engine-out climb-angle check are single conditions, not
        integrated segments.
    """

    name: str
    subsystem: str
    kind: PhaseKind
    in_takeoff: bool
    is_reserve: bool
    integrates_fuel: bool


def _phase(name, kind, *, subsystem=None, in_takeoff, is_reserve, integrates_fuel):
    """Build a :class:`PhaseSpec`, defaulting the subsystem name to the phase name."""
    return PhaseSpec(name, subsystem or name, kind, in_takeoff, is_reserve, integrates_fuel)


MISSION_PHASES: tuple[PhaseSpec, ...] = (
    # -------------- Balanced-field takeoff (14 CFR 25.113) --------------
    _phase("v0v1", PhaseKind.GROUND_ROLL, in_takeoff=True, is_reserve=False, integrates_fuel=False),
    _phase("v1vr", PhaseKind.GROUND_ROLL, in_takeoff=True, is_reserve=False, integrates_fuel=False),
    _phase("v1v0", PhaseKind.GROUND_ROLL, in_takeoff=True, is_reserve=False, integrates_fuel=False),
    _phase("rotate", PhaseKind.ROTATION, in_takeoff=True, is_reserve=False, integrates_fuel=False),
    # -------------- One-engine-inoperative climb gradient (14 CFR 25.121) --------------
    _phase(
        "EngineOutClimbAngle",
        PhaseKind.CLIMB_ANGLE,
        subsystem="engineoutclimb",
        in_takeoff=True,
        is_reserve=False,
        integrates_fuel=False,
    ),
    # -------------- Design mission --------------
    _phase("climb", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=False, integrates_fuel=True),
    _phase("cruise", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=False, integrates_fuel=True),
    _phase("descent", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=False, integrates_fuel=True),
    # -------------- Part 25 reserves --------------
    _phase("reserve_climb", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=True, integrates_fuel=True),
    _phase("reserve_cruise", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=True, integrates_fuel=True),
    _phase("reserve_descent", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=True, integrates_fuel=True),
    _phase("loiter", PhaseKind.STEADY_FLIGHT, in_takeoff=False, is_reserve=True, integrates_fuel=True),
)
"""Every phase ``FullMissionWithReserve`` builds, in the order fuel accumulates through them.

``loiter`` is last, which is why the total fuel used by the whole mission including reserves
is read from the loiter phase's integrator.
"""

PHASE_SPECS_BY_NAME: dict[str, PhaseSpec] = {spec.name: spec for spec in MISSION_PHASES}


def phase_supplied_inputs(phase: str | PhaseSpec) -> VariableSet:
    """Return the variables OpenConcept supplies to the aircraft model in a phase.

    These are the promoted names an aircraft model may consume. Anything outside this set
    is either a design parameter under ``ac|`` (supplied from the top level) or is not
    available, in which case consuming it creates an unconnected input silently holding a
    default.

    Parameters
    ----------
    phase : str or PhaseSpec
        Phase name, or the specification itself.

    Returns
    -------
    VariableSet
        The supplied flight conditions and control inputs for that phase.

    Raises
    ------
    KeyError
        If ``phase`` names a phase ``FullMissionWithReserve`` does not build.
    """
    spec = PHASE_SPECS_BY_NAME[phase] if isinstance(phase, str) else phase

    supplied = [*_ATMOSPHERIC_CONDITIONS, *_CONTROL_INPUTS]

    if spec.kind is PhaseKind.GROUND_ROLL:
        # On the ground: vertical speed is zero and braking friction applies.
        supplied += [*_KINEMATIC_CONDITIONS, *_GROUND_ROLL_ONLY, _TIME]
    elif spec.kind is PhaseKind.STEADY_FLIGHT:
        supplied += [*_KINEMATIC_CONDITIONS, _TIME]
    elif spec.kind is PhaseKind.ROTATION:
        # Rotation uses Raymer's circular-arc transition, not an integrated ODE, so it has
        # no groundspeed or flight-path-angle vectors to offer.
        supplied += [_TIME]
    elif spec.kind is PhaseKind.CLIMB_ANGLE:
        # A single flight condition at V2, not an integrated segment: no duration.
        supplied += [Variable("fltcond|cosgamma", None, vectorized=True, description="Cosine of flight path angle")]

    return VariableSet(supplied)


# ==============================================================================
# What cdadt sets on the mission
# ==============================================================================
MISSION_PROFILE_INPUTS = VariableSet(
    [
        Variable("takeoff|h", "ft", description="Takeoff and landing field elevation"),
        Variable("cruise|h0", "ft", description="Initial cruise altitude"),
        Variable("mission_range", "NM", description="Design range flown in the main mission"),
        Variable("payload", "lbm", description="Mission payload"),
        Variable("reserve_range", "NM", description="Range flown in the reserve mission"),
        Variable("reserve|h0", "ft", description="Reserve mission cruise altitude"),
        Variable("loiter|h0", "ft", description="Loiter altitude"),
        Variable("loiter_duration", "s", description="Loiter duration"),
    ]
)
"""Mission profile parameters ``FullMissionWithReserve`` exposes as ``IndepVarComp`` outputs.

Declared in ``openconcept/mission/profiles.py`` lines 45-54. These are set on the built
problem, not connected, because OpenConcept creates them as independent variables.
"""


# ==============================================================================
# What cdadt reads back out
# ==============================================================================
@dataclass(frozen=True)
class MissionOutput:
    """One result cdadt reads out of the mission black box.

    Parameters
    ----------
    name : str
        The cdadt-facing name. Stable even if the OpenConcept path changes.
    path : str
        Variable path relative to the mission group, e.g.
        ``"loiter.fuel_burn_integ.fuel_burn_final"``.
    units : str or None
        Units to read the value in.
    description : str
        What the result means and, where relevant, which regulation consumes it.
    """

    name: str
    path: str
    units: str | None
    description: str


MISSION_OUTPUTS: tuple[MissionOutput, ...] = (
    MissionOutput(
        "block_fuel",
        "descent.fuel_burn_integ.fuel_burn_final",
        "kg",
        "Fuel burned over the design mission, excluding reserves. The usual sizing objective.",
    ),
    MissionOutput(
        "total_fuel",
        "loiter.fuel_burn_integ.fuel_burn_final",
        "kg",
        "Fuel burned over the design mission plus reserves and loiter. This closes the "
        "MTOW = OEW + payload + fuel sizing loop, because loiter is the last phase fuel "
        "accumulates through.",
    ),
    MissionOutput(
        "takeoff_field_length",
        "bfl.distance_continue",
        "ft",
        "Balanced field length: distance to clear the obstacle after an engine failure at "
        "V1. Constrained by 14 CFR 25.113.",
    ),
    MissionOutput(
        "accelerate_stop_distance",
        "bfl.distance_abort",
        "ft",
        "Distance to accelerate to V1 and stop. Equal to the continue distance by "
        "construction, since OpenConcept solves V1 implicitly to make them equal; reading "
        "both is how that solve is verified rather than assumed.",
    ),
    MissionOutput(
        "decision_speed",
        "takeoff|v1",
        "kn",
        "V1, solved implicitly so that the continue and abort distances match.",
    ),
    MissionOutput(
        "rotation_speed",
        "v1vr.takeoff|vr",
        "kn",
        "V_R, rotation speed, set as a multiple of the stall speed in takeoff configuration.",
    ),
    MissionOutput(
        "engine_out_climb_gradient",
        "engineoutclimb.gamma",
        "rad",
        "Climb angle at V2 with one engine inoperative. Constrained by 14 CFR 25.121.",
    ),
    MissionOutput(
        "stall_speed_takeoff",
        "v0v1.Vstall_eas",
        "kn",
        "Stall speed in takeoff configuration at MTOW.",
    ),
    MissionOutput(
        "mission_range_flown",
        "descent.ode_integ_phase.range_final",
        "NM",
        "Range actually flown by the end of descent. Should equal the requested design "
        "range; comparing them verifies the cruise-duration balance converged.",
    ),
    MissionOutput(
        "reserve_range_flown",
        "reserve_descent.ode_integ_phase.range_final",
        "NM",
        "Cumulative range at the end of the reserve descent.",
    ),
)
"""Every result cdadt reads from the mission, declared once.

Paths are relative to the mission group so that the group can be added under any name.
"""

MISSION_OUTPUTS_BY_NAME: dict[str, MissionOutput] = {output.name: output for output in MISSION_OUTPUTS}

THROTTLE_BY_PHASE_PATH = "{phase}.throttle"
"""Format string for a phase's throttle vector, constrained for thrust margin.

Kept as a format string rather than enumerated, because it applies to every phase and the
phase list already lives in :data:`MISSION_PHASES`.
"""
