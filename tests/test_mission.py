"""Unit tests for :mod:`cdadt.mission`."""

from __future__ import annotations

import numpy as np
import pytest

from cdadt import MissionProfile, PhaseSchedule
from cdadt.mission import STEADY_FLIGHT_PHASES

pytestmark = pytest.mark.unit


def test_two_endpoints_interpolate_across_the_nodes():
    """Two values mean a linear schedule, which is how every phase is written."""
    schedule = PhaseSchedule(equivalent_airspeed=[230, 252], vertical_speed=[2300, 400])
    assert schedule.airspeed_at(5) == pytest.approx(np.linspace(230, 252, 5))
    assert schedule.vertical_speed_at(5) == pytest.approx(np.linspace(2300, 400, 5))


def test_one_value_is_held_constant():
    """A cruise vertical speed of zero should not have to be written eleven times."""
    assert PhaseSchedule([252], [0]).airspeed_at(3) == pytest.approx([252, 252, 252])


def test_a_full_length_schedule_is_used_as_given():
    """An arbitrary profile passes through untouched."""
    values = [200, 210, 250]
    assert PhaseSchedule(values, values).airspeed_at(3) == pytest.approx(values)


def test_a_wrong_length_schedule_raises():
    """Broadcasting would quietly fly a mission nobody asked for."""
    with pytest.raises(ValueError, match="4 values but the mission has 11 nodes"):
        PhaseSchedule([1, 2, 3, 4], [0, 0]).airspeed_at(11)


def test_every_steady_flight_phase_must_be_scheduled():
    """OpenConcept has no default profile; an unscheduled phase flies a placeholder."""
    schedules = {phase: PhaseSchedule([250], [0]) for phase in STEADY_FLIGHT_PHASES if phase != "loiter"}
    with pytest.raises(ValueError, match="loiter"):
        MissionProfile(parameters={}, schedules=schedules)


def test_the_b738_profile_loads_with_its_continuation_schedule(profile):
    """The shipped case carries the design mission and the ladder up to it."""
    assert profile.parameters["mission_range"] == (2800.0, "nmi")
    assert profile.parameters["cruise|h0"] == (35000.0, "ft")
    assert set(profile.schedules) == set(STEADY_FLIGHT_PHASES)
    assert len(profile.continuation) == 2
    # The first step must be strictly easier than the design mission, or it is not continuation.
    assert profile.continuation[0].parameters["mission_range"][0] < profile.parameters["mission_range"][0]
    assert profile.continuation[0].parameters["cruise|h0"][0] < profile.parameters["cruise|h0"][0]


def test_a_half_specified_schedule_override_raises():
    """Overriding Ueas alone silently keeps a vs nobody wrote down for that step."""
    with pytest.raises(ValueError, match="both 'Ueas' and 'vs' are required"):
        MissionProfile.from_dict(
            {
                "schedule": {phase: {"Ueas": {"value": 250}, "vs": {"value": 0}} for phase in STEADY_FLIGHT_PHASES},
                "continuation": [{"description": "bad", "schedule": {"climb": {"Ueas": {"value": 250}}}}],
            }
        )


def test_continuation_steps_keep_their_order(profile):
    """The ladder is climbed in the order it is written."""
    assert [step.description for step in profile.continuation] == [
        "short range at low altitude, shallow descent",
        "design range and altitude, descent rate still shallow",
    ]
