"""Unit and integration: the files a study leaves behind.

The unit tests run against a fake box that publishes a known trajectory, so that what is being
tested is the reading, the discovery and the plotting rather than OpenConcept. The integration
tests run against a real converged box, because "discovers the phases of the real mission" is a
claim about the real mission and cannot be made against a fake.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from cdadt import ArtifactError, MissionTrajectory, StudyArtifacts, Trace

#: The phases and the traces a fake box publishes, in flight order.
PHASES = ("climb", "cruise", "descent")


class FakeBox:
    """A box that publishes a known trajectory, and nothing else.

    Mirrors :class:`~cdadt.blackbox.OpenConceptSizingBox` where it is used: ``readable`` is a
    *method* returning a name-sorted mapping, exactly as the real one is. That is not a detail --
    an earlier version of this double exposed it as a property and in name order, and both
    differences hid a real defect that only the integration tests below caught.

    Parameters
    ----------
    phases : sequence of str, optional
        Phases to publish, in flight order. Default :data:`PHASES`.
    missing : sequence of str, optional
        Trace names to withhold from the *last* phase, so a partly published quantity can be
        tested without inventing a second fake.
    ground_roll : sequence of str, optional
        Subsystems that publish the abscissa but are not steady flight phases, as the real box's
        takeoff phases do. Default: one, named so that name order would place it first.
    built : bool, optional
        What :attr:`is_built` reports. Default ``True``.
    """

    def __init__(self, phases=PHASES, missing=(), ground_roll=("aaa_v1v0",), built=True):
        self.is_built = bool(built)
        self.problem = object()
        self._values = {}
        traces = [MissionTrajectory.ABSCISSA, *MissionTrajectory.TRACES]
        for index, phase in enumerate(phases):
            self._values[f"mission.{phase}.{MissionTrajectory.PHASE_MARKER}"] = np.array([index + 1.0])
            for trace in traces:
                if trace.name in missing and phase == phases[-1]:
                    continue
                self._values[f"mission.{phase}.{trace.name}"] = np.linspace(index, index + 1, 5)
        # Ground roll: publishes the abscissa and the traces, but not the phase marker.
        for phase in ground_roll:
            for trace in traces:
                self._values[f"mission.{phase}.{trace.name}"] = np.zeros(5)
        self._values["ac|weights|MTOW"] = np.array([78345.0])

    def readable(self):
        """Every path this box publishes, sorted by name as the real box sorts them."""
        return dict.fromkeys(sorted(self._values))

    def has(self, name):
        """Whether the box publishes ``name``."""
        return name in self._values

    def get(self, name, units=None):
        """Return a published value. Units are recorded as read, not converted."""
        return self._values[name]


# =============================================================================================
# Trace
# =============================================================================================


@pytest.mark.unit
def test_a_trace_must_name_a_variable():
    """An unnamed trace would address the phase group itself."""
    with pytest.raises(ValueError, match="must name a variable"):
        Trace("", "ft", "Altitude")


@pytest.mark.unit
def test_a_trace_reports_what_it_was_given_and_where_it_lives():
    """Including under a non-default mission path, which is what makes it addressable at all."""
    trace = Trace("fltcond|h", "ft", "Altitude (ft)")
    assert (trace.name, trace.units, trace.label) == ("fltcond|h", "ft", "Altitude (ft)")
    assert trace.path("mission", "climb") == "mission.climb.fltcond|h"
    assert trace.path("profile", "climb") == "profile.climb.fltcond|h"
    assert repr(trace) == "Trace('fltcond|h', 'ft')"


# =============================================================================================
# MissionTrajectory
# =============================================================================================


@pytest.mark.unit
def test_the_phases_are_discovered_and_ordered_by_where_they_start():
    """Discovery, not a hardcoded list: a different mission model must plot without an edit."""
    trajectory = MissionTrajectory(FakeBox())
    assert trajectory.phases == PHASES
    assert repr(trajectory) == "MissionTrajectory(3 phases, 7 traces)"


@pytest.mark.unit
def test_a_ground_roll_phase_is_not_part_of_the_trajectory():
    """It publishes the abscissa but not the phase marker, and the rejected takeoff doubles back.

    The fake names it so that name order would put it first, which is what makes this a test of
    the discriminator rather than of the sort.
    """
    trajectory = MissionTrajectory(FakeBox())
    assert "aaa_v1v0" not in trajectory.phases


@pytest.mark.unit
def test_alphabetical_order_is_not_flight_order():
    """The real mission flies loiter last, after three reserve phases that sort before it."""
    flown = ("climb", "reserve_climb", "loiter")
    trajectory = MissionTrajectory(FakeBox(phases=flown))

    assert trajectory.phases == flown
    assert trajectory.phases != tuple(sorted(flown)), "the order under test is the sorted one"


@pytest.mark.unit
def test_a_box_with_no_steady_phase_says_so_rather_than_plotting_nothing():
    """An empty plot is indistinguishable from a mission nobody flew."""
    with pytest.raises(ArtifactError, match="no steady flight phase under 'mission'"):
        MissionTrajectory(FakeBox(phases=(), ground_roll=()))


@pytest.mark.unit
def test_the_mission_path_is_honoured():
    """The mission subsystem is named in the case file and need not be called ``mission``."""
    with pytest.raises(ArtifactError, match="no steady flight phase under 'profile'"):
        MissionTrajectory(FakeBox(), mission_path="profile")


@pytest.mark.unit
def test_a_trace_one_phase_is_missing_is_dropped_rather_than_drawn_with_a_hole():
    """A trajectory drawn across a gap reads as a trajectory, not as missing data."""
    trajectory = MissionTrajectory(FakeBox(missing=("fltcond|CL",)))

    available = [trace.name for trace in trajectory.available]
    assert "fltcond|CL" not in available
    assert "fltcond|h" in available
    assert len(trajectory) == len(MissionTrajectory.TRACES) - 1


@pytest.mark.unit
def test_reading_a_trace_no_phase_publishes_names_the_phase():
    """The escape hatch from the silent drop above: asked for explicitly, it must explain."""
    trajectory = MissionTrajectory(FakeBox(missing=("fltcond|CL",)))
    with pytest.raises(ArtifactError, match=re.escape("Phase 'descent' does not publish 'fltcond|CL'")):
        trajectory.read(Trace("fltcond|CL", None, "CL"))


@pytest.mark.unit
def test_a_trace_is_concatenated_across_the_phases_in_flight_order():
    """One vector per mission, not one per phase, and in the order they are flown."""
    trajectory = MissionTrajectory(FakeBox())
    values = trajectory.read(MissionTrajectory.ABSCISSA)

    assert values.shape == (15,)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(3.0)
    assert np.all(np.diff(values) >= 0.0), "the phases were not concatenated in flight order"


@pytest.mark.unit
def test_iterating_yields_every_plottable_trace_against_the_abscissa():
    """What the figure is drawn from, and the same length every time."""
    trajectory = MissionTrajectory(FakeBox())
    drawn = list(trajectory)

    assert len(drawn) == len(MissionTrajectory.TRACES)
    for trace, abscissa, values in drawn:
        assert isinstance(trace, Trace)
        assert abscissa.shape == values.shape


# =============================================================================================
# StudyArtifacts
# =============================================================================================


@pytest.mark.unit
def test_the_directory_is_created_including_parents(tmp_path):
    """A caller naming an output directory should not have to create it first."""
    artifacts = StudyArtifacts(tmp_path / "study" / "run1")
    assert artifacts.directory.is_dir()
    assert repr(artifacts).startswith("StudyArtifacts(")


@pytest.mark.unit
def test_text_and_json_are_written_and_reported(tmp_path):
    """The report as printed, and the numbers as data, in one place."""
    artifacts = StudyArtifacts(tmp_path)
    artifacts.write_text("the report", "report.txt")
    artifacts.write_json({"MTOW": 78345.6435}, "results.json")

    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == "the report"
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["MTOW"] == 78345.6435
    assert [path.name for path in artifacts.written()] == ["report.txt", "results.json"]


@pytest.mark.unit
def test_an_unbuilt_box_cannot_be_diagrammed(tmp_path):
    """There is no model yet, and the message says which method was skipped."""
    artifacts = StudyArtifacts(tmp_path)
    with pytest.raises(ArtifactError, match="has not been built"):
        artifacts.write_n2(FakeBox(built=False))


@pytest.mark.unit
def test_the_trajectory_is_plotted_from_the_fake_box(tmp_path):
    """Every panel drawn, the file written, and no interactive backend touched."""
    artifacts = StudyArtifacts(tmp_path)
    written = artifacts.write_trajectory(FakeBox(), title="fake", name="trajectory.pdf")

    assert written == tmp_path / "trajectory.pdf"
    assert written.stat().st_size > 0


@pytest.mark.unit
def test_an_odd_number_of_panels_leaves_the_spare_axes_blank(tmp_path):
    """Seven traces in two columns is eight panels; the eighth must not be an empty grid."""
    artifacts = StudyArtifacts(tmp_path)
    written = artifacts.write_trajectory(FakeBox(missing=("fltcond|CL", "fltcond|M")), title="fake", columns=3)
    assert written.stat().st_size > 0


# =============================================================================================
# Against the real box
# =============================================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_the_real_mission_publishes_the_phases_and_traces_this_module_expects(converged_analysis):
    """The claim that matters: the discovery finds OpenConcept's own seven steady phases.

    Written against the live model rather than a list, so a change in the mission model shows up
    here as a failure rather than as a plot that quietly lost a leg.
    """
    trajectory = MissionTrajectory(converged_analysis.box, mission_path=converged_analysis.config.mission_path)

    # In flight order, which is neither name order nor the order the box lists them in: the
    # diversion is flown after the design mission, and the loiter last of all.
    assert trajectory.phases == (
        "climb",
        "cruise",
        "descent",
        "reserve_climb",
        "reserve_cruise",
        "reserve_descent",
        "loiter",
    )
    assert "v1v0" not in trajectory.phases, "the rejected takeoff is not part of the flown trajectory"
    assert trajectory.available == MissionTrajectory.TRACES, "the real box no longer publishes every plotted quantity"


@pytest.mark.integration
@pytest.mark.slow
def test_the_ground_roll_is_excluded_by_being_published_and_not_by_being_addressable(converged_analysis):
    """Discovery must read ``readable``; ``has`` answers a different question and the wrong one.

    On the real model these two disagree, and they disagree precisely on the phases that have to
    be excluded: ``v0v1``, ``v1vr`` and ``v1v0`` all answer ``has`` for the marker while
    publishing no such output. A refactor that swapped one for the other would put the rejected
    takeoff back into the figure, doubling it back along the range axis -- and it would still
    look like a trajectory. Only ``rotate`` is excluded by both, so only this test would notice.
    """
    box = converged_analysis.box
    mission = converged_analysis.config.mission_path
    published = set(box.readable())

    for phase in ("v0v1", "v1vr", "v1v0"):
        marker = f"{mission}.{phase}.{MissionTrajectory.PHASE_MARKER}"
        assert f"{mission}.{phase}.range" in published, f"'{phase}' should still publish the abscissa"
        assert marker not in published, f"'{phase}' now publishes the phase marker and would be plotted"
        assert box.has(marker), f"'{phase}' no longer distinguishes 'readable' from 'has', so this test is moot"

    for phase in ("climb", "loiter"):
        assert f"{mission}.{phase}.{MissionTrajectory.PHASE_MARKER}" in published


@pytest.mark.integration
@pytest.mark.slow
def test_the_fuel_panel_is_cumulative_mission_fuel_and_not_per_phase(converged_analysis):
    """The fuel integrator carries over between phases, so the panel is the mission total.

    Locked because it decides what the panel means, and because the two readings are
    indistinguishable at the first phase: both start near zero. Checked where they differ --
    every later phase begins exactly where the one before ended, and the last node is
    ``total_fuel``.
    """
    box = converged_analysis.box
    mission = converged_analysis.config.mission_path
    trajectory = MissionTrajectory(box, mission_path=mission)

    ends = [
        np.asarray(box.get(f"{mission}.{phase}.fuel_burn_integ.fuel_burn", units="lbm"), dtype=float).reshape(-1)
        for phase in trajectory.phases
    ]
    for phase, previous, current in zip(trajectory.phases[1:], ends[:-1], ends[1:], strict=True):
        assert current[0] == pytest.approx(
            previous[-1], rel=1e-12
        ), f"'{phase}' does not carry on from the phase before"

    # The response catalogue defines total_fuel as this same integrator's final value, so the
    # panel ending on the mission total is an identity rather than a coincidence -- and this
    # asserts the identity still holds through the units the panel is read in.
    last = trajectory.phases[-1]
    total = float(
        np.asarray(box.get(f"{mission}.{last}.fuel_burn_integ.fuel_burn_final", units="lbm"), dtype=float).reshape(-1)[
            0
        ]
    )
    assert ends[-1][-1] == pytest.approx(total, rel=1e-9), "the last node is not the mission's total fuel"


@pytest.mark.integration
@pytest.mark.slow
def test_the_real_trajectory_is_monotonic_in_range_and_plots(tmp_path, converged_analysis):
    """Range must increase across the whole mission, or the phases were assembled out of order."""
    box = converged_analysis.box
    trajectory = MissionTrajectory(box, mission_path=converged_analysis.config.mission_path)
    distance = trajectory.read(MissionTrajectory.ABSCISSA)

    assert np.all(np.diff(distance) >= -1e-6), "range does not increase monotonically along the mission"

    artifacts = StudyArtifacts(tmp_path)
    assert artifacts.write_trajectory(box, title="b738").stat().st_size > 0
    assert artifacts.write_n2(box).stat().st_size > 0
