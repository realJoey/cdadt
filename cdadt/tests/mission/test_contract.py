"""Verification of the cdadt/OpenConcept interface contract.

Claim class: ``contract``. Every declaration in :mod:`cdadt.mission.contract` is checked
against a really-built ``FullMissionWithReserve``, into which OpenConcept has instantiated a
real cdadt aircraft model. Nothing is mocked and nothing is checked by reading source.

These are the highest-value tests in the repository. The failure they exist to catch is
silent: OpenConcept promotes the aircraft model with ``promotes_inputs=["*"]``, so a name
cdadt gets wrong does not raise -- it becomes an unconnected input holding its declared
default, and the mission converges to a number computed from it. When OpenConcept changes a
variable name, this file fails with the name that moved.

Checks are exhaustive by construction: each test iterates the full declared set and asserts
on every entry, collecting all failures before reporting. Spot-checking a few names would
produce a gate that verifies names it happened to pick.
"""

from __future__ import annotations

import openmdao.api as om
import pytest
from openconcept.mission import FullMissionWithReserve
from openmdao.utils.units import unit_conversion

from cdadt.mission.aircraft_model import AircraftModelFactory
from cdadt.mission.contract import (
    AIRCRAFT_MODEL_OUTPUTS,
    MISSION_OUTPUTS,
    MISSION_PHASES,
    MISSION_PROFILE_INPUTS,
    PhaseKind,
    phase_supplied_inputs,
)
from cdadt.tests.mission.support import simple_disciplines

pytestmark = pytest.mark.contract

NUM_NODES = 3


@pytest.fixture(scope="module")
def built_problem():
    """Return a set-up problem containing a real mission with a real cdadt aircraft model.

    Module-scoped: setting up the full mission is the expensive part, and every test here
    inspects the same built model rather than running it.
    """
    model_class = AircraftModelFactory(simple_disciplines()).build()

    prob = om.Problem()
    prob.model.add_subsystem(
        "mission",
        FullMissionWithReserve(num_nodes=NUM_NODES, aircraft_model=model_class),
        promotes_inputs=["ac|*"],
    )
    prob.setup(check=False)
    prob.final_setup()
    return prob


def _promoted_names(system, io_type):
    """Return the set of promoted variable names of one type within ``system``."""
    metadata = system.get_io_metadata(io_type, metadata_keys=["units"], return_rel_names=False)
    return {meta["prom_name"] for meta in metadata.values()}


def _phase_system(problem, spec):
    """Return the OpenMDAO system for a declared phase."""
    return problem.model.mission._get_subsystem(spec.subsystem)


# ==============================================================================
# Phases
# ==============================================================================
def test_every_declared_phase_exists_in_the_built_mission(built_problem):
    """Each phase in MISSION_PHASES is a real subsystem of the mission group."""
    missing = [spec.name for spec in MISSION_PHASES if _phase_system(built_problem, spec) is None]
    assert not missing, (
        f"MISSION_PHASES declares phases that FullMissionWithReserve does not build: {missing}. "
        f"Mission subsystems: {sorted(s.name for s, _ in built_problem.model.mission._subsystems_allprocs.values())}"
    )


def test_no_built_phase_is_missing_from_the_declaration(built_problem):
    """Every phase OpenConcept builds is declared, so nothing is silently unmodeled.

    The mission also contains ``missionparams``, ``bfl`` and ``resrange``, which are not
    phases and do not instantiate an aircraft model; those are excluded by requiring the
    subsystem to contain an ``acmodel``.
    """
    declared = {spec.subsystem for spec in MISSION_PHASES}
    built_with_acmodel = set()
    for subsystem, _ in built_problem.model.mission._subsystems_allprocs.values():
        if isinstance(subsystem, om.Group) and subsystem._get_subsystem("acmodel") is not None:
            built_with_acmodel.add(subsystem.name)

    assert built_with_acmodel == declared, (
        f"Phases that instantiate an aircraft model but are not declared: "
        f"{sorted(built_with_acmodel - declared)}; declared but not built: {sorted(declared - built_with_acmodel)}"
    )


@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_aircraft_model_is_instantiated_with_the_declared_flight_phase(built_problem, spec):
    """OpenConcept passes the declared ``flight_phase`` string to the aircraft model.

    Providers branch on this string to decide the aircraft's configuration -- flaps and
    gear for takeoff, clean for cruise -- so a mismatch would silently run the wrong
    configuration rather than raise.
    """
    acmodel = _phase_system(built_problem, spec)._get_subsystem("acmodel")
    assert acmodel.options["flight_phase"] == spec.name


@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_aircraft_model_receives_the_declared_node_count(built_problem, spec):
    """Vectorized variables are sized by the node count OpenConcept passes.

    The engine-out climb-angle check is a single flight condition, so OpenConcept builds it
    with one node regardless of the mission's node count.
    """
    acmodel = _phase_system(built_problem, spec)._get_subsystem("acmodel")
    expected = 1 if spec.kind is PhaseKind.CLIMB_ANGLE else NUM_NODES
    assert acmodel.options["num_nodes"] == expected


# ==============================================================================
# What the aircraft model must produce
# ==============================================================================
@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_required_outputs_are_produced_in_every_phase(built_problem, spec):
    """thrust, drag and mass exist as promoted outputs of the aircraft model in each phase."""
    produced = _promoted_names(_phase_system(built_problem, spec)._get_subsystem("acmodel"), "output")
    missing = sorted(AIRCRAFT_MODEL_OUTPUTS.names - produced)
    assert not missing, f"Phase '{spec.name}' aircraft model does not produce {missing}. Produced: {sorted(produced)}"


@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_required_output_units_match_the_declaration(built_problem, spec):
    """Declared units agree with what OpenMDAO assigned, so no conversion is implied.

    A units mismatch here would be silently converted by OpenMDAO at the connection, which
    is exactly the kind of error that produces a plausible wrong answer.
    """
    acmodel = _phase_system(built_problem, spec)._get_subsystem("acmodel")
    metadata = acmodel.get_io_metadata("output", metadata_keys=["units"], return_rel_names=False)
    units_by_name = {meta["prom_name"]: meta["units"] for meta in metadata.values()}

    mismatches = {
        variable.name: (variable.units, units_by_name[variable.name])
        for variable in AIRCRAFT_MODEL_OUTPUTS
        if units_by_name.get(variable.name) != variable.units
    }
    assert not mismatches, f"Phase '{spec.name}': declared units differ from the built model: {mismatches}"


# ==============================================================================
# What OpenConcept supplies
# ==============================================================================
@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_every_declared_supplied_input_exists_in_the_phase(built_problem, spec):
    """Each variable declared as supplied really is available at the phase level.

    Over-declaring here is the dangerous direction: a provider that consumes a name
    OpenConcept does not supply gets an unconnected input at its default. This test is what
    makes ``phase_supplied_inputs`` trustworthy enough for
    :class:`~cdadt.mission.aircraft_model.AircraftModelFactory` to validate against.
    """
    phase = _phase_system(built_problem, spec)
    available = _promoted_names(phase, "input") | _promoted_names(phase, "output")

    missing = sorted(name for name in phase_supplied_inputs(spec).names if name not in available)
    assert not missing, (
        f"Phase '{spec.name}' does not supply {missing}, but the contract declares it does. "
        f"A provider consuming one of these would hold an unconnected input at its default."
    )


@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_supplied_input_units_are_dimensionally_identical_to_the_declaration(built_problem, spec):
    """Supplied flight conditions carry units physically identical to those declared.

    The property that matters is that reading a supplied variable in cdadt's declared units
    implies no conversion -- a declared ``K`` against OpenConcept's ``degK``, or
    ``N * m**-2`` against ``N/m**2``, is the same quantity spelled differently and is
    correct. A declared ``ft`` against an actual ``m`` is not, and OpenMDAO would silently
    convert it into a plausible wrong answer.

    So this compares conversion factors rather than strings: a factor of exactly 1 with no
    offset means the two spellings denote the same unit.
    """
    phase = _phase_system(built_problem, spec)
    units_by_name = {}
    for io_type in ("input", "output"):
        for meta in phase.get_io_metadata(io_type, metadata_keys=["units"], return_rel_names=False).values():
            units_by_name.setdefault(meta["prom_name"], meta["units"])

    mismatches = {}
    for variable in phase_supplied_inputs(spec):
        actual = units_by_name.get(variable.name)
        if actual == variable.units:
            continue
        if actual is None or variable.units is None:
            mismatches[variable.name] = (variable.units, actual, "one is dimensionless")
            continue
        try:
            factor, offset = unit_conversion(actual, variable.units)
        except Exception as err:
            mismatches[variable.name] = (variable.units, actual, f"incompatible: {err}")
            continue
        if factor != pytest.approx(1.0, rel=1e-12) or offset != pytest.approx(0.0, abs=1e-12):
            mismatches[variable.name] = (variable.units, actual, f"factor={factor}, offset={offset}")

    assert not mismatches, f"Phase '{spec.name}': declared vs actual units {mismatches}"


# ==============================================================================
# What cdadt reads back
# ==============================================================================
def test_every_declared_mission_output_resolves(built_problem):
    """Each path in MISSION_OUTPUTS resolves to a real variable, in its declared units.

    This is the test that catches an OpenConcept version bump. Every path is attempted and
    all failures are reported together, so one moved variable does not hide the next.
    """
    failures = {}
    for output in MISSION_OUTPUTS:
        try:
            built_problem.get_val(f"mission.{output.path}", units=output.units)
        except Exception as err:
            failures[output.name] = f"{output.path} ({output.units}): {type(err).__name__}: {err}"

    assert not failures, "Declared mission outputs that do not resolve:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in sorted(failures.items())
    )


def test_every_declared_profile_input_resolves(built_problem):
    """Each mission profile parameter exists and accepts its declared units."""
    failures = {}
    for variable in MISSION_PROFILE_INPUTS:
        try:
            built_problem.get_val(f"mission.{variable.name}", units=variable.units)
        except Exception as err:
            failures[variable.name] = f"{type(err).__name__}: {err}"

    assert not failures, "Declared mission profile inputs that do not resolve:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in sorted(failures.items())
    )


def test_throttle_resolves_for_every_phase_that_has_one(built_problem):
    """Throttle is readable per phase, since the thrust-margin requirement constrains it."""
    failures = {}
    for spec in MISSION_PHASES:
        try:
            built_problem.get_val(f"mission.{spec.subsystem}.throttle")
        except Exception as err:
            failures[spec.name] = f"{type(err).__name__}: {err}"

    assert not failures, "Phases whose throttle does not resolve:\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in sorted(failures.items())
    )


def test_balanced_field_length_reads_both_continue_and_abort(built_problem):
    """Both balanced-field distances are declared, not just the one used as a constraint.

    OpenConcept solves V1 implicitly so the two are equal. Reading both is how that solve
    is verified after a run rather than assumed.
    """
    declared = {output.name for output in MISSION_OUTPUTS}
    assert {"takeoff_field_length", "accelerate_stop_distance"} <= declared


def test_total_fuel_is_read_from_the_last_phase_fuel_accumulates_through():
    """Total mission fuel comes from loiter, the final phase in the accumulation chain.

    Reading it from an earlier phase would understate the fuel that closes the sizing loop,
    and would still converge.
    """
    accumulating = [spec for spec in MISSION_PHASES if spec.integrates_fuel]
    assert accumulating[-1].name == "loiter"

    total_fuel = next(output for output in MISSION_OUTPUTS if output.name == "total_fuel")
    assert total_fuel.path.startswith("loiter.")
