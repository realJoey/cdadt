"""Validation: cdadt's B738 sizing against OpenConcept's own B738 sizing example.

This is the test that justifies the word "black box". Both models are run in the same process,
in the same environment, on the same aircraft and the same mission, and every top-level result
is compared. If cdadt's assembly of OpenConcept's components differed from OpenConcept's own
assembly of them anywhere that matters, these numbers would not agree.

The reference is *run*, not quoted. A table of numbers pasted from a previous session tests
that nobody edited the table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cdadt import AircraftDefinition, MissionProfile, SizingAnalysis

CASES = Path(__file__).resolve().parent.parent / "cases"

#: Result name in cdadt, mapped to the variable path in OpenConcept's example and the units to
#: compare in. Every top-level quantity the reference example prints appears here.
COMPARISONS: dict[str, tuple[str, str | None]] = {
    "MTOW": ("ac|weights|MTOW", "kg"),
    "OEW": ("ac|weights|OEW", "kg"),
    "MLW": ("ac|weights|MLW", "kg"),
    "block_fuel": ("mission.descent.fuel_burn_integ.fuel_burn_final", "kg"),
    "total_fuel": ("mission.loiter.fuel_burn_integ.fuel_burn_final", "kg"),
    "takeoff_field_length": ("mission.bfl.distance_continue", "ft"),
    "abort_distance": ("mission.bfl.distance_abort", "ft"),
    "V1": ("mission.takeoff|v1", "kn"),
    "V2": ("mission.engineoutclimb.takeoff|v2", "kn"),
    "engine_out_climb_angle": ("mission.engineoutclimb.gamma", "rad"),
    "CLmax_cruise": ("ac|aero|CLmax_cruise", None),
    "CLmax_TO": ("ac|aero|CLmax_TO", None),
    "wing_MAC": ("ac|geom|wing|MAC", "m"),
    "hstab_S_ref": ("ac|geom|hstab|S_ref", "m**2"),
    "vstab_S_ref": ("ac|geom|vstab|S_ref", "m**2"),
}

NUM_NODES = 21  # what OpenConcept's run_738_sizing_analysis uses


@pytest.fixture(scope="module")
def cdadt_results() -> dict[str, float]:
    """Run cdadt's sizing analysis on the shipped B738 case."""
    analysis = SizingAnalysis(
        aircraft=AircraftDefinition.from_yaml(CASES / "b738_aircraft.yaml"),
        profile=MissionProfile.from_yaml(CASES / "b738_mission.yaml"),
        num_nodes=NUM_NODES,
    )
    return analysis.run()


@pytest.fixture(scope="module")
def openconcept_problem():
    """Run OpenConcept's own B738 sizing example and return its converged problem."""
    from openconcept.examples.B738_sizing import run_738_sizing_analysis

    return run_738_sizing_analysis(num_nodes=NUM_NODES)


@pytest.mark.validation
@pytest.mark.slow
def test_every_top_level_result_matches_the_reference_example(cdadt_results, openconcept_problem):
    """cdadt reproduces OpenConcept's published B738 sizing, quantity by quantity.

    The tolerance is 1e-6 relative, which is the Newton solver's own convergence, not an
    engineering tolerance. The two models are solving the same equations; anything looser
    would hide a real difference.
    """
    mismatches = []
    for name, (path, units) in COMPARISONS.items():
        reference = openconcept_problem.get_val(path, units=units).item()
        computed = cdadt_results[name]
        # A relative test on a reference of zero is a division by zero, not a loose test.
        error = abs(computed) if reference == 0.0 else abs(computed - reference) / abs(reference)
        if error >= 1e-6:
            mismatches.append(f"{name}: cdadt {computed!r} vs OpenConcept {reference!r}")

    assert not mismatches, "cdadt does not reproduce the reference:\n" + "\n".join(mismatches)


@pytest.mark.validation
@pytest.mark.slow
def test_the_balanced_field_is_actually_balanced(cdadt_results):
    """V1 is solved so that continuing and aborting cover the same distance. Check it did."""
    assert cdadt_results["takeoff_field_length"] == pytest.approx(cdadt_results["abort_distance"], rel=1e-6)


@pytest.mark.validation
@pytest.mark.slow
def test_the_weight_loop_is_closed(cdadt_results):
    """MTOW must equal OEW plus payload plus total fuel, or the loop did not converge."""
    total = cdadt_results["OEW"] + cdadt_results["payload"] + cdadt_results["total_fuel"]
    assert cdadt_results["MTOW"] == pytest.approx(total, rel=1e-9)


@pytest.mark.validation
@pytest.mark.slow
def test_reserves_are_carried(cdadt_results):
    """Total fuel must exceed block fuel, or the aircraft was sized without its reserves."""
    assert cdadt_results["total_fuel"] > cdadt_results["block_fuel"]
