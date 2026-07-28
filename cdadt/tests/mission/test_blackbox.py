"""Tests for :mod:`cdadt.mission.blackbox`.

Claim classes: ``unit`` for the profile objects and path construction, ``integration`` for
the run that actually converges a mission.

The integration test is the one that matters. It builds a real ``FullMissionWithReserve``
with a real cdadt aircraft model, runs the continuation schedule, and asserts on physical
relationships that a broken wiring would violate: fuel accumulating monotonically across
phases, the balanced-field continue and abort distances matching, and the flown range
closing on the requested range. It makes no claim about *correct* physics -- the fixture
model is deliberately crude -- only that the black box is driven and read correctly. Physical
correctness is established separately against OpenConcept's B738 example.
"""

from __future__ import annotations

import itertools

import numpy as np
import openmdao.api as om
import pytest

from cdadt.mission.aircraft_model import AircraftModelFactory
from cdadt.mission.blackbox import (
    ContinuationStep,
    MissionBlackBox,
    MissionProfile,
    PhaseSchedule,
)
from cdadt.mission.contract import MISSION_OUTPUTS, MISSION_PHASES
from cdadt.tests.mission.support import simple_disciplines


def _schedules():
    """Return an airspeed/vertical-speed schedule for every steady-flight phase."""
    return {
        "climb": PhaseSchedule([230.0, 252.0], [2300.0, 400.0]),
        "cruise": PhaseSchedule([252.0, 252.0], [0.0, 0.0]),
        "descent": PhaseSchedule([252.0, 250.0], [-800.0, -800.0]),
        "reserve_climb": PhaseSchedule([230.0, 230.0], [3000.0, 2300.0]),
        "reserve_cruise": PhaseSchedule([250.0, 250.0], [0.0, 0.0]),
        "reserve_descent": PhaseSchedule([250.0, 250.0], [-800.0, -800.0]),
        "loiter": PhaseSchedule([250.0, 250.0], [0.0, 0.0]),
    }


def _parameters(mission_range=2000.0, cruise_altitude=33000.0, reserve_altitude=15000.0):
    """Return a full set of mission profile parameters."""
    return {
        "takeoff|h": (0.0, "ft"),
        "cruise|h0": (cruise_altitude, "ft"),
        "mission_range": (mission_range, "NM"),
        "payload": (18e3, "kg"),
        "reserve_range": (200.0, "NM"),
        "reserve|h0": (reserve_altitude, "ft"),
        "loiter|h0": (1500.0, "ft"),
        "loiter_duration": (30.0 * 60.0, "s"),
    }


def _profile():
    """Return a mission profile with a continuation schedule from an easy mission."""
    return MissionProfile(
        schedules=_schedules(),
        parameters=_parameters(),
        continuation=[
            ContinuationStep(
                "short range at low altitude",
                overrides={
                    "mission_range": (500.0, "NM"),
                    "cruise|h0": (5000.0, "ft"),
                    "reserve|h0": (1000.0, "ft"),
                    "reserve_range": (100.0, "NM"),
                },
            ),
        ],
    )


# ==============================================================================
# MissionProfile and PhaseSchedule
# ==============================================================================
@pytest.mark.unit
def test_schedule_resamples_two_endpoints_across_the_nodes():
    """A two-point schedule is interpolated, so a profile is independent of node count."""
    schedule = PhaseSchedule([200.0, 300.0], [1000.0, 0.0])
    assert schedule.airspeed_at_nodes(5) == pytest.approx([200.0, 225.0, 250.0, 275.0, 300.0])
    assert schedule.vertical_speed_at_nodes(3) == pytest.approx([1000.0, 500.0, 0.0])


@pytest.mark.unit
def test_schedule_accepts_a_full_length_vector():
    """A schedule already at the node count is used as given."""
    values = [200.0, 210.0, 260.0]
    assert PhaseSchedule(values, [0.0, 0.0, 0.0]).airspeed_at_nodes(3) == pytest.approx(values)


@pytest.mark.unit
def test_schedule_rejects_a_wrong_length_vector():
    """Broadcasting the wrong length would fly a different profile than the one requested."""
    with pytest.raises(ValueError, match="has 4 values but the mission has 3 nodes"):
        PhaseSchedule([1.0, 2.0, 3.0, 4.0], [0.0, 0.0]).airspeed_at_nodes(3)


@pytest.mark.unit
def test_profile_requires_a_schedule_for_every_steady_flight_phase():
    """An unscheduled phase would fly OpenConcept's placeholder defaults."""
    incomplete = _schedules()
    del incomplete["loiter"]
    with pytest.raises(ValueError, match=r"\['loiter'\]"):
        MissionProfile(schedules=incomplete, parameters=_parameters())


@pytest.mark.unit
def test_profile_requires_every_mission_parameter():
    """Mission parameters have no defaults, for the same reason configuration entries do not."""
    incomplete = _parameters()
    del incomplete["reserve_range"]
    with pytest.raises(ValueError, match=r"\['reserve_range'\]"):
        MissionProfile(schedules=_schedules(), parameters=incomplete)


# ==============================================================================
# Construction and paths
# ==============================================================================
@pytest.mark.unit
def test_even_node_count_is_rejected():
    """Simpson's rule needs 2N+1 points; an even count would silently misintegrate."""
    model_class = AircraftModelFactory(simple_disciplines()).build()
    with pytest.raises(ValueError, match="must be odd"):
        MissionBlackBox(model_class, _profile(), num_nodes=10)


@pytest.mark.unit
def test_declared_outputs_are_exposed():
    """The black box advertises exactly the results declared in the contract."""
    model_class = AircraftModelFactory(simple_disciplines()).build()
    blackbox = MissionBlackBox(model_class, _profile(), num_nodes=3)
    assert blackbox.outputs == MISSION_OUTPUTS


@pytest.mark.unit
def test_paths_are_built_from_the_group_name_not_hardcoded():
    """Renaming the mission subsystem moves every path with it."""
    model_class = AircraftModelFactory(simple_disciplines()).build()
    blackbox = MissionBlackBox(model_class, _profile(), num_nodes=3)

    prob = om.Problem()
    blackbox.build(prob.model, name="sizing_mission")

    assert blackbox.path("total_fuel").startswith("sizing_mission.")
    assert blackbox.throttle_path("cruise") == "sizing_mission.cruise.throttle"


@pytest.mark.unit
def test_throttle_path_uses_the_subsystem_name_for_the_engine_out_check():
    """The OEI climb phase is added as ``engineoutclimb`` while its flight_phase differs.

    Using the flight_phase string as a path would silently fail to resolve.
    """
    model_class = AircraftModelFactory(simple_disciplines()).build()
    blackbox = MissionBlackBox(model_class, _profile(), num_nodes=3)
    assert blackbox.throttle_path("EngineOutClimbAngle") == "mission.engineoutclimb.throttle"


@pytest.mark.unit
def test_unknown_output_name_raises_listing_what_is_available():
    """A typo'd result name fails loudly rather than returning nothing."""
    model_class = AircraftModelFactory(simple_disciplines()).build()
    blackbox = MissionBlackBox(model_class, _profile(), num_nodes=3)
    with pytest.raises(KeyError, match="is not a declared mission output"):
        blackbox.path("blockfuel")


# ==============================================================================
# Running the mission
# ==============================================================================
@pytest.fixture(scope="module")
def converged_mission():
    """Build, converge, and return a real mission with a real cdadt aircraft model.

    Module-scoped because converging the mission is the expensive part of this file.
    """
    model_class = AircraftModelFactory(simple_disciplines()).build()
    blackbox = MissionBlackBox(model_class, _profile(), num_nodes=3)

    prob = om.Problem()
    blackbox.build(prob.model)

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, atol=1e-8, rtol=1e-8)
    prob.model.nonlinear_solver.options["iprint"] = -1
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
    prob.model.nonlinear_solver.linesearch.options["iprint"] = -1
    prob.model.linear_solver = om.DirectSolver()

    prob.setup(check=False)
    prob.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    prob.set_val("ac|weights|MTOW", 79e3, units="kg")
    prob.set_val("ac|aero|CLmax_TO", 2.2)

    blackbox.converge(prob)
    return blackbox, prob


@pytest.mark.integration
def test_the_mission_converges_and_every_declared_output_is_finite(converged_mission):
    """Reading the full declared output set returns finite numbers after a run."""
    blackbox, prob = converged_mission
    results = blackbox.read_outputs(prob)

    assert set(results) == {output.name for output in MISSION_OUTPUTS}
    non_finite = {name: value for name, value in results.items() if not np.all(np.isfinite(value))}
    assert not non_finite, f"Mission outputs that are not finite: {non_finite}"


@pytest.mark.integration
def test_fuel_accumulates_monotonically_across_the_phases(converged_mission):
    """Fuel burned only ever increases, and the reserve mission adds to the block fuel.

    This is what verifies OpenConcept's cross-phase state linking reached the cdadt
    integrator. If it had not, each phase would restart from zero and the total fuel that
    closes the sizing loop would be the loiter fuel alone.
    """
    blackbox, prob = converged_mission
    results = blackbox.read_outputs(prob)

    fuel_by_phase = [
        prob.get_val(f"mission.{spec.subsystem}.fuel_burn_integ.fuel_burn_final", units="kg").item()
        for spec in MISSION_PHASES
        if spec.integrates_fuel
    ]

    assert all(
        later >= earlier for earlier, later in itertools.pairwise(fuel_by_phase)
    ), f"Fuel burn is not monotonic across phases: {fuel_by_phase}"
    assert results["total_fuel"].item() > results["block_fuel"].item(), (
        "Total fuel including reserves is not greater than block fuel, so the reserve "
        "mission did not accumulate onto the design mission."
    )


@pytest.mark.integration
def test_the_balanced_field_length_is_balanced(converged_mission):
    """Continue and abort distances match, which is what V1 is solved to achieve.

    OpenConcept solves V1 implicitly so these are equal. Asserting it verifies the implicit
    solve converged rather than assuming it did, and is the reason both distances are
    declared as outputs instead of only the one used as a constraint.
    """
    blackbox, prob = converged_mission
    results = blackbox.read_outputs(prob)

    continue_distance = results["takeoff_field_length"].item()
    abort_distance = results["accelerate_stop_distance"].item()

    assert continue_distance > 0.0
    assert abort_distance == pytest.approx(continue_distance, rel=1e-4), (
        f"V1 solve did not balance the field length: continue {continue_distance:.1f} ft "
        f"vs abort {abort_distance:.1f} ft"
    )


@pytest.mark.integration
def test_the_flown_range_closes_on_the_requested_range(converged_mission):
    """Distance flown by end of descent equals the requested design range.

    The cruise duration is solved by a BalanceComp against this target, so a mismatch means
    that balance did not converge -- which otherwise shows up only as a subtly wrong fuel
    burn.
    """
    blackbox, prob = converged_mission
    results = blackbox.read_outputs(prob)

    requested, units = blackbox.profile.parameters["mission_range"]
    assert units == "NM"
    assert results["mission_range_flown"].item() == pytest.approx(requested, rel=1e-4)


@pytest.mark.integration
def test_the_engine_out_climb_gradient_is_available_and_positive(converged_mission):
    """The 25.121 climb angle is read from the mission, and the aircraft climbs on one engine."""
    blackbox, prob = converged_mission
    gamma = blackbox.read_outputs(prob)["engine_out_climb_gradient"].item()
    assert gamma > 0.0, "Engine-out climb gradient is not positive; the model cannot satisfy 25.121"


@pytest.mark.integration
def test_the_decision_speed_lies_below_the_rotation_speed(converged_mission):
    """V1 <= V_R, as it must be for the takeoff decision to be physically meaningful."""
    blackbox, prob = converged_mission
    results = blackbox.read_outputs(prob)
    assert results["decision_speed"].item() <= results["rotation_speed"].item() + 1e-6


@pytest.mark.integration
def test_setting_the_profile_reaches_every_scheduled_phase(converged_mission):
    """The airspeed and vertical-speed vectors OpenConcept flies are the ones requested."""
    blackbox, prob = converged_mission
    mismatches = {}
    for phase_name, schedule in blackbox.profile.schedules.items():
        actual = prob.get_val(f"mission.{phase_name}.fltcond|Ueas", units=schedule.airspeed_units)
        expected = schedule.airspeed_at_nodes(blackbox.num_nodes)
        if not np.allclose(actual, expected):
            mismatches[phase_name] = (expected, actual)
    assert not mismatches, f"Phases flying a different airspeed schedule than requested: {mismatches}"
