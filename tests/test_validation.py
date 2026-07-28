"""Validation: cdadt's sizing run against OpenConcept's own, executed live.

This is the test that justifies the word "black box". OpenConcept's ``run_738_sizing_analysis``
is imported and *run*, in this process, in this environment, at the same grid, and every
quantity cdadt reports is compared against the reference problem it produced.

The reference is run, not quoted. A table of numbers pasted from a previous session tests only
that nobody edited the table.

What a match proves, and what it does not
-----------------------------------------

It proves that cdadt's case file, continuation ladder, parameter routing and result reading
reproduce the reference exactly: same aircraft, same mission, same convergence. Because cdadt
drives OpenConcept's own sizing group rather than a re-assembly of its parts, the physics is
identical by construction, and this test is what turns "by construction" into "checked". It
would fail if the case file transcribed a number wrong, if the continuation ladder converged
somewhere else, if a response were read from the wrong path or in the wrong units, or if a
future change to cdadt started perturbing the box.

It proves nothing about whether OpenConcept's model is right. See :doc:`/validation`.
"""

from __future__ import annotations

import pytest

from cdadt import SizingAnalysis

#: cdadt result name -> (path in the reference problem, units to compare in). Every scalar the
#: reference example prints appears here, plus the weights and geometry it computes.
COMPARISONS: dict[str, tuple[str, str | None]] = {
    "MTOW": ("ac|weights|MTOW", "kg"),
    "OEW": ("ac|weights|OEW", "kg"),
    "MLW": ("ac|weights|MLW", "kg"),
    "payload": ("ac|weights|W_payload", "kg"),
    "block_fuel": ("mission.descent.fuel_burn_integ.fuel_burn_final", "kg"),
    "total_fuel": ("mission.loiter.fuel_burn_integ.fuel_burn_final", "kg"),
    "takeoff_field_length": ("mission.bfl.distance_continue", "ft"),
    "abort_distance": ("mission.bfl.distance_abort", "ft"),
    "V1": ("mission.takeoff|v1", "kn"),
    "V2": ("mission.engineoutclimb.takeoff|v2", "kn"),
    "engine_out_climb_gradient": ("mission.engineoutclimb.gamma", "rad"),
    "CLmax_cruise": ("ac|aero|CLmax_cruise", None),
    "CLmax_takeoff": ("ac|aero|CLmax_TO", None),
    "wing_MAC": ("ac|geom|wing|MAC", "m"),
    "hstab_area": ("ac|geom|hstab|S_ref", "m**2"),
    "vstab_area": ("ac|geom|vstab|S_ref", "m**2"),
    "structure_weight": ("empty_weight.W_structure", "kg"),
    "wing_weight": ("empty_weight.W_wing", "kg"),
    "fuselage_weight": ("empty_weight.W_fuselage", "kg"),
    "engines_weight": ("empty_weight.W_engines", "kg"),
    "fuselage_wetted_area": ("ac|geom|fuselage|S_wet", "m**2"),
    "mission_range_flown": ("mission.descent.ode_integ_phase.range_final", "nmi"),
}

#: The Newton solver's own convergence, not an engineering tolerance. The two runs solve the
#: same equations in the same code; anything looser would hide a real difference.
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def reference_problem(converged_analysis: SizingAnalysis):
    """Run OpenConcept's own B738 sizing example at cdadt's grid and return its problem."""
    from openconcept.examples.B738_sizing import run_738_sizing_analysis

    return run_738_sizing_analysis(num_nodes=converged_analysis.box.num_nodes)


@pytest.mark.validation
@pytest.mark.slow
def test_every_reported_quantity_matches_the_reference(converged_analysis, reference_problem):
    """cdadt reproduces OpenConcept's published B738 sizing, quantity by quantity."""
    results = converged_analysis.results()
    mismatches = []
    for name, (path, units) in COMPARISONS.items():
        reference = reference_problem.get_val(path, units=units).item()
        computed = float(results[name])
        # A relative test against a reference of zero is a division by zero, not a loose test.
        error = abs(computed) if reference == 0.0 else abs(computed - reference) / abs(reference)
        if error >= TOLERANCE:
            mismatches.append(f"  {name}: cdadt {computed!r} vs OpenConcept {reference!r} (rel {error:.3e})")

    assert not mismatches, "cdadt does not reproduce the reference:\n" + "\n".join(mismatches)


@pytest.mark.validation
@pytest.mark.slow
def test_the_reference_and_cdadt_agree_on_the_throttle_history(converged_analysis, reference_problem):
    """Vector results match too, not only the scalars a summary table would show."""
    results = converged_analysis.results()
    for phase in ("climb", "cruise", "descent"):
        reference = reference_problem.get_val(f"mission.{phase}.throttle")
        computed = results[f"{phase}_throttle"]
        assert computed == pytest.approx(reference, rel=TOLERANCE), f"{phase} throttle history differs"


@pytest.mark.validation
@pytest.mark.slow
def test_the_balanced_field_is_actually_balanced(converged_analysis):
    """V1 is solved so that continuing and aborting cover the same distance. Check it did.

    A converged balanced-field solve makes these equal by construction. A run where they differ
    has not converged, whatever else it reports.
    """
    results = converged_analysis.results()
    assert results["takeoff_field_length"] == pytest.approx(results["abort_distance"], rel=TOLERANCE)


@pytest.mark.validation
@pytest.mark.slow
def test_the_weight_loop_is_closed(converged_analysis):
    """MTOW must equal OEW plus payload plus total fuel, or the sizing loop did not close."""
    results = converged_analysis.results()
    total = results["OEW"] + results["payload"] + results["total_fuel"]
    assert results["MTOW"] == pytest.approx(total, rel=1e-9)


@pytest.mark.validation
@pytest.mark.slow
def test_reserves_are_carried(converged_analysis):
    """Total fuel must exceed block fuel, or the aircraft was sized without its reserves."""
    results = converged_analysis.results()
    assert results["total_fuel"] > results["block_fuel"]


@pytest.mark.validation
@pytest.mark.slow
def test_the_mission_actually_flown_is_the_one_that_was_asked_for(converged_analysis):
    """The range flown must be the design range, not whatever the solver settled on."""
    results = converged_analysis.results()
    requested = converged_analysis.config.initial_conditions().values["mission_range"][0]
    assert results["mission_range_flown"] == pytest.approx(requested, rel=1e-6)
