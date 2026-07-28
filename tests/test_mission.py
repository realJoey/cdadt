"""Unit: the initial conditions, and the ladder that converges the hard ones.

Nothing here builds a model. Conditions are written into anything that accepts ``set``,
``shape_of`` and ``run``, which is what makes them testable without OpenConcept -- and is why
:class:`~cdadt.mission.InitialConditions` is written against that protocol rather than against
an OpenMDAO problem.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from cdadt import ContinuationLadder, ContinuationStep, InitialConditions, MissionError


class RecordingBox:
    """Records everything written to it, and how many times it was converged."""

    def __init__(self, shapes: dict[str, tuple[int, ...]] | None = None) -> None:
        self.values: dict[str, tuple] = {}
        self.runs = 0
        self._shapes = shapes or {}

    def set(self, name, value, units=None):
        """Record a written value."""
        self.values[name] = (value, units)

    def shape_of(self, name):
        """Return the declared shape, defaulting to a scalar."""
        return self._shapes.get(name, (1,))

    def has(self, name):
        """Return whether the fake publishes that name."""
        return name in self._shapes

    def run(self):
        """Count a convergence."""
        self.runs += 1


SHAPES = {
    "mission.mission_range": (1,),
    "mission.cruise|h0": (1,),
    "mission.climb.fltcond|Ueas": (5,),
    "ac|weights|MTOW": (1,),
}


# =============================================================================================
# Resolving names
# =============================================================================================


@pytest.mark.unit
def test_a_name_is_tried_as_written_then_under_the_mission_path():
    """Names are written as ``set_values`` writes them; the box nests the mission one level."""
    box = RecordingBox(SHAPES)
    conditions = InitialConditions({}, mission_path="mission")

    assert conditions.resolve("ac|weights|MTOW", box) == "ac|weights|MTOW"
    assert conditions.resolve("cruise|h0", box) == "mission.cruise|h0"
    assert conditions.resolve("climb.fltcond|Ueas", box) == "mission.climb.fltcond|Ueas"


@pytest.mark.unit
def test_a_name_that_resolves_neither_way_names_both_attempts():
    """Whether it is ``cruise|h0`` or ``mission.cruise|h0`` is what a failure must answer."""
    conditions = InitialConditions({"nonsense": (1.0, None)})
    with pytest.raises(MissionError, match=re.escape("neither 'nonsense' nor 'mission.nonsense'")):
        conditions.check(RecordingBox(SHAPES))


@pytest.mark.unit
def test_a_custom_mission_path_is_honoured():
    """The mission subsystem is named by the case file, not assumed."""
    box = RecordingBox({"trajectory.cruise|h0": (1,)})
    conditions = InitialConditions({}, mission_path="trajectory")
    assert conditions.mission_path == "trajectory"
    assert conditions.resolve("cruise|h0", box) == "trajectory.cruise|h0"


# =============================================================================================
# Resampling
# =============================================================================================


@pytest.mark.unit
def test_a_scalar_is_written_as_a_scalar():
    """OpenMDAO broadcasts it; there is nothing to resample."""
    assert InitialConditions({}).resample(35000.0, (1,), "cruise|h0") == 35000.0


@pytest.mark.unit
def test_two_values_are_interpolated_across_the_target():
    """Exactly what ``np.linspace(2300.0, 400.0, num_nodes)`` does in the reference run script."""
    assert InitialConditions({}).resample([230, 270], (5,), "climb.fltcond|Ueas") == pytest.approx(
        np.array([230.0, 240.0, 250.0, 260.0, 270.0])
    )


@pytest.mark.unit
def test_one_value_per_node_is_used_as_given():
    """The third accepted form."""
    values = [1.0, 2.0, 3.0]
    assert InitialConditions({}).resample(values, (3,), "x") == pytest.approx(np.asarray(values))


@pytest.mark.unit
def test_a_single_element_list_is_broadcast():
    """A list of one is a constant written the long way."""
    assert InitialConditions({}).resample([7.0], (3,), "x") == pytest.approx(np.full(3, 7.0))


@pytest.mark.unit
def test_a_wrong_length_schedule_is_refused():
    """Broadcasting it would quietly fly a different mission than the case file asks for."""
    with pytest.raises(MissionError, match="given 3 values but the black box declares 5"):
        InitialConditions({}).resample([1.0, 2.0, 3.0], (5,), "climb.fltcond|vs")


# =============================================================================================
# Applying
# =============================================================================================


@pytest.mark.unit
def test_applying_writes_every_condition_resolved_and_resampled():
    """The whole job: resolve the name, shape the value, keep the units."""
    box = RecordingBox(SHAPES)
    InitialConditions(
        {
            "mission_range": (2800.0, "nmi"),
            "climb.fltcond|Ueas": ([230, 252], "kn"),
            "ac|weights|MTOW": (50000.0, "kg"),
        }
    ).apply(box)

    assert box.values["mission.mission_range"] == (2800.0, "nmi")
    assert box.values["ac|weights|MTOW"] == (50000.0, "kg")
    written, units = box.values["mission.climb.fltcond|Ueas"]
    assert units == "kn"
    assert written == pytest.approx(np.linspace(230, 252, 5))


@pytest.mark.unit
def test_conditions_behave_as_a_collection():
    """Membership, iteration and length are how a report walks them."""
    conditions = InitialConditions({"a": (1.0, None), "b": (2.0, "kg")})
    assert "a" in conditions and "c" not in conditions
    assert list(iter(conditions)) == ["a", "b"]
    assert len(conditions) == 2
    assert conditions.values["b"] == (2.0, "kg")
    assert repr(conditions) == "InitialConditions(2 values)"


@pytest.mark.unit
def test_merging_writes_the_override_on_top():
    """How a rung is applied: design conditions first, then what it relaxes."""
    design = InitialConditions({"mission_range": (2800.0, "nmi"), "cruise|h0": (35000.0, "ft")})
    merged = design.merged_with({"mission_range": (500.0, "nmi")})

    assert merged.values["mission_range"] == (500.0, "nmi")
    assert merged.values["cruise|h0"] == (35000.0, "ft")
    # The original is untouched, so the next rung starts from the design conditions again.
    assert design.values["mission_range"] == (2800.0, "nmi")
    assert merged.merged_with(design).values["mission_range"] == (2800.0, "nmi")


# =============================================================================================
# The ladder
# =============================================================================================


@pytest.mark.unit
def test_a_rung_exposes_what_it_overrides():
    """The run log prints the description; a report prints what each rung relaxed."""
    step = ContinuationStep("short range", {"mission_range": (500.0, "nmi")})
    assert step.description == "short range"
    assert set(step.conditions) == {"mission_range"}
    assert repr(step) == "ContinuationStep('short range')"


@pytest.mark.unit
def test_converging_walks_every_rung_and_then_the_design_mission():
    """One solve per rung, plus one for the design mission itself."""
    box = RecordingBox(SHAPES)
    conditions = InitialConditions({"mission_range": (2800.0, "nmi")})
    ladder = ContinuationLadder(
        [
            ContinuationStep("first", {"mission_range": (500.0, "nmi")}),
            ContinuationStep("second", {"mission_range": (1500.0, "nmi")}),
        ]
    )
    ladder.converge(box, conditions)

    assert box.runs == 3
    assert len(ladder) == 2
    assert [step.description for step in ladder] == ["first", "second"]
    assert ladder.steps[0].description == "first"
    # The design mission is written last, so the box ends holding what was asked for.
    assert box.values["mission.mission_range"] == (2800.0, "nmi")
    assert repr(ladder) == "ContinuationLadder(2 steps)"


@pytest.mark.unit
def test_an_empty_ladder_converges_the_design_mission_directly():
    """Legal, and for a short mission it works; for a long one it does not."""
    box = RecordingBox(SHAPES)
    ContinuationLadder().converge(box, InitialConditions({"mission_range": (2800.0, "nmi")}))
    assert box.runs == 1


@pytest.mark.unit
def test_converging_verbosely_narrates_each_rung(capsys):
    """The run log is how a long continuation is watched; it must name the steps."""
    ladder = ContinuationLadder([ContinuationStep("short range at low altitude")])
    ladder.converge(RecordingBox(SHAPES), InitialConditions({}), verbose=True)

    printed = capsys.readouterr().out
    assert "short range at low altitude" in printed
    assert "continuation complete" in printed


@pytest.mark.unit
def test_converging_quietly_prints_nothing(capsys):
    """The default, because an optimizer walks this once per study and not per iteration."""
    ContinuationLadder([ContinuationStep("a rung")]).converge(RecordingBox(SHAPES), InitialConditions({}))
    assert capsys.readouterr().out == ""


# =============================================================================================
# The analysis that owns them
# =============================================================================================


@pytest.mark.integration
def test_a_sizing_analysis_exposes_the_pieces_it_composes(converged_analysis):
    """Its performance discipline, its box and its ladder are what a report and a test read."""
    performance = converged_analysis.performance
    assert len(performance.conditions) == 56
    assert len(performance.ladder) == 2
    assert repr(converged_analysis) == (
        "SizingAnalysis('openconcept.examples.B738_sizing:B738SizingMissionAnalysis', num_nodes=21)"
    )
    assert repr(performance).startswith("Performance(InitialConditions(")
