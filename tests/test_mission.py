"""Unit: the mission value objects, and the continuation walk that uses them.

Nothing here builds a model. A :class:`~cdadt.mission.MissionProfile` is applied to anything
that accepts ``set(name, value, units)`` and ``run()``, which is exactly what makes it testable
without OpenConcept -- and is why the class is written against that protocol rather than against
an OpenMDAO problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdadt import ContinuationStep, MissionProfile, PhaseSchedule
from cdadt.mission import GROUND_ROLL_PHASES, STEADY_FLIGHT_PHASES


class RecordingBox:
    """Records everything written to it, and how many times it was converged."""

    def __init__(self) -> None:
        self.values: dict[str, tuple] = {}
        self.runs = 0

    def set(self, name, value, units=None):
        """Record a written value."""
        self.values[name] = (value, units)

    def run(self):
        """Count a convergence."""
        self.runs += 1


def _profile(**kwargs) -> MissionProfile:
    """Return a complete profile, with every steady-flight phase scheduled."""
    return MissionProfile(
        parameters=kwargs.pop("parameters", {"mission_range": (2800.0, "nmi")}),
        schedules=kwargs.pop("schedules", {phase: PhaseSchedule(250.0, 0.0) for phase in STEADY_FLIGHT_PHASES}),
        **kwargs,
    )


# =============================================================================================
# Schedules
# =============================================================================================


@pytest.mark.unit
def test_a_schedule_given_one_value_per_node_is_used_as_given():
    """The third accepted form, alongside a constant and a pair of endpoints."""
    values = [230.0, 240.0, 250.0, 260.0, 270.0]
    schedule = PhaseSchedule(values, values)
    assert schedule.airspeed_at(5) == pytest.approx(np.asarray(values))
    assert schedule.vertical_speed_at(5) == pytest.approx(np.asarray(values))


@pytest.mark.unit
def test_a_schedule_reports_its_units_and_reprs_as_both_of_them():
    """Both schedules carry units independently; a repr that hid one would mislead."""
    schedule = PhaseSchedule([230, 252], [2300, 400], airspeed_units="kn", vertical_speed_units="ft/min")
    assert schedule.airspeed_units == "kn"
    assert schedule.vertical_speed_units == "ft/min"
    text = repr(schedule)
    assert "kn" in text and "ft/min" in text


# =============================================================================================
# The profile
# =============================================================================================


@pytest.mark.unit
def test_a_profile_exposes_everything_it_was_built_from():
    """A profile that could not be read back could not be reported or archived."""
    step = ContinuationStep("easier", {"mission_range": (500.0, "nmi")}, {"descent": PhaseSchedule(250.0, -800.0)})
    profile = _profile(continuation=[step], takeoff_speed_guess=(110.0, "kn"), mission_path="trajectory")

    assert profile.parameters == {"mission_range": (2800.0, "nmi")}
    assert set(profile.schedules) == set(STEADY_FLIGHT_PHASES)
    assert profile.takeoff_speed_guess == (110.0, "kn")
    assert profile.mission_path == "trajectory"
    assert set(iter(profile)) == set(STEADY_FLIGHT_PHASES)
    assert repr(profile) == "MissionProfile(7 phases, 1 continuation steps)"


@pytest.mark.unit
def test_a_continuation_step_exposes_what_it_overrides():
    """The run log prints the description; a report prints what each rung relaxed."""
    step = ContinuationStep("short range", {"mission_range": (500.0, "nmi")}, {"descent": PhaseSchedule(250.0, -800.0)})
    assert step.description == "short range"
    assert step.parameters == {"mission_range": (500.0, "nmi")}
    assert set(step.schedules) == {"descent"}
    assert repr(step) == "ContinuationStep('short range')"


@pytest.mark.unit
def test_the_mission_path_prefixes_every_parameter_written():
    """The box publishes mission parameters under its mission subsystem, not at the top."""
    box = RecordingBox()
    _profile(mission_path="trajectory").apply(box, num_nodes=5)
    assert "trajectory.mission_range" in box.values
    assert "trajectory.cruise.fltcond|Ueas" in box.values


@pytest.mark.unit
def test_the_ground_roll_phases_are_seeded_and_the_steady_ones_scheduled():
    """Ground roll integrates from a standstill, so it needs a guess rather than a profile."""
    box = RecordingBox()
    profile = _profile(takeoff_speed_guess=(100.0, "kn"))
    profile.seed_takeoff_speeds(box, num_nodes=5)

    for phase in GROUND_ROLL_PHASES:
        value, units = box.values[f"mission.{phase}.fltcond|Utrue"]
        assert units == "kn"
        assert value == pytest.approx(np.full(5, 100.0))
    assert not any("fltcond|Ueas" in name for name in box.values)


# =============================================================================================
# Converging
# =============================================================================================


@pytest.mark.unit
def test_converging_walks_every_rung_and_then_the_design_mission():
    """One solve per rung, plus one for the design mission itself."""
    box = RecordingBox()
    profile = _profile(
        continuation=[
            ContinuationStep("first", {"mission_range": (500.0, "nmi")}),
            ContinuationStep("second", {"mission_range": (1500.0, "nmi")}),
        ]
    )
    profile.converge(box, num_nodes=5)

    assert box.runs == 3
    # The design mission is written last, so the box ends holding what was asked for.
    assert box.values["mission.mission_range"] == (2800.0, "nmi")


@pytest.mark.unit
def test_converging_verbosely_narrates_each_rung(capsys):
    """The run log is how a long continuation is watched; it must name the steps."""
    profile = _profile(continuation=[ContinuationStep("short range at low altitude")])
    profile.converge(RecordingBox(), num_nodes=5, verbose=True)

    printed = capsys.readouterr().out
    assert "short range at low altitude" in printed
    assert "continuation complete" in printed


@pytest.mark.unit
def test_converging_quietly_prints_nothing(capsys):
    """The default, because an optimizer walks this once per study and not per iteration."""
    _profile(continuation=[ContinuationStep("a rung")]).converge(RecordingBox(), num_nodes=5)
    assert capsys.readouterr().out == ""


# =============================================================================================
# The analysis object that owns a profile
# =============================================================================================


@pytest.mark.integration
def test_a_sizing_analysis_exposes_the_pieces_it_composes(converged_analysis):
    """Its performance discipline, its box and its grid are what a report and a test read."""
    assert converged_analysis.performance.num_nodes == converged_analysis.box.num_nodes
    assert set(converged_analysis.performance.profile) == set(STEADY_FLIGHT_PHASES)
    assert repr(converged_analysis) == (
        "SizingAnalysis('openconcept.examples.B738_sizing:B738SizingMissionAnalysis', num_nodes=21)"
    )
