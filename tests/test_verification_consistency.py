"""Verification: internal consistency. Do the reported quantities agree with each other?

Every result cdadt prints is read out of the black box independently, by the discipline that
owns it, from its own path and in its own units. Nothing forces them to be mutually consistent,
so whether they are is a real check rather than a tautology -- and it is the check that catches
the two errors most likely to survive everything else: a response wired to the wrong path, and a
response declared in the wrong units.

A mass read in pounds and reported as kilograms is off by 2.2. It would still be positive, still
vary sensibly with the design, still reproduce run to run, and still be reported to four decimal
places. What it would *not* do is add up.

The identities below are stated in physics, not in the box's internals. Where an identity would
only hold because of a modelling choice inside OpenConcept -- maximum landing weight being a
fixed fraction of takeoff weight, say -- the check is written as an inequality that any sane
aeroplane satisfies rather than pinning a constant this repository does not own.
"""

from __future__ import annotations

import pytest

#: Responses whose declared units are re-read in a second unit to confirm the declaration is
#: honoured, as ``response -> (alternate unit, conversion factor from the declared unit)``.
UNIT_ROUND_TRIPS = {
    "MTOW": ("lb", 2.2046226218),
    "OEW": ("lb", 2.2046226218),
    "total_fuel": ("lb", 2.2046226218),
    "takeoff_field_length": ("m", 0.3048),
    "V1": ("m/s", 0.5144444444),
    "engine_out_climb_gradient": ("deg", 57.29577951308232),
    "hstab_area": ("ft**2", 10.763910416709722),
}


@pytest.fixture(scope="module")
def results(converged_analysis):
    """The converged results of the shipped sizing case."""
    return converged_analysis.results()


# =============================================================================================
# Units
# =============================================================================================


@pytest.mark.verification
@pytest.mark.parametrize("name", sorted(UNIT_ROUND_TRIPS))
def test_each_response_is_read_in_the_units_it_declares(converged_analysis, results, name):
    """Reading the same quantity in a second unit must give exactly the converted value.

    This is what proves :class:`~cdadt.parameters.Response` units are applied on the way out
    rather than merely recorded. A response declared ``kg`` but read in the box's native pounds
    would pass every magnitude check in this repository and fail here.
    """
    alternate, factor = UNIT_ROUND_TRIPS[name]
    catalog = converged_analysis.catalog
    box = converged_analysis.box

    declared = float(results[name])
    converted = float(box.get(catalog.path(name), units=alternate))
    assert converted == pytest.approx(
        declared * factor, rel=1e-9
    ), f"{name}: {declared} in {catalog.units(name)} is not {converted} in {alternate}"


# =============================================================================================
# Mass identities
# =============================================================================================


@pytest.mark.verification
@pytest.mark.slow
def test_the_takeoff_weight_is_the_sum_of_its_parts(results):
    """The closure the whole model exists to solve, checked on the reported numbers."""
    total = float(results["OEW"]) + float(results["payload"]) + float(results["total_fuel"])
    assert float(results["MTOW"]) == pytest.approx(total, rel=1e-9)


@pytest.mark.verification
@pytest.mark.slow
def test_the_weights_are_ordered_the_way_an_aeroplane_requires(results):
    """Empty < landing < takeoff, and the payload and fuel both fit inside the difference."""
    empty, landing, takeoff = float(results["OEW"]), float(results["MLW"]), float(results["MTOW"])
    assert empty < landing < takeoff, f"OEW {empty}, MLW {landing}, MTOW {takeoff} are not ordered"
    assert float(results["payload"]) < takeoff - empty
    assert float(results["total_fuel"]) < takeoff - empty


@pytest.mark.verification
@pytest.mark.slow
def test_the_structural_mass_is_a_part_of_the_empty_mass(results):
    """Primary structure must be a fraction of operating empty weight, not a multiple of it."""
    structure, empty = float(results["structure_weight"]), float(results["OEW"])
    assert 0.0 < structure < empty
    assert 0.3 < structure / empty < 0.9


@pytest.mark.verification
@pytest.mark.slow
def test_the_engine_masses_are_mutually_consistent(results, converged_analysis):
    """Total installed engine mass must be the single-engine mass times the engine count.

    Both are read from the box independently, in the same units, by the same discipline. If one
    were converted and the other not, this is where it would show.
    """
    count = float(converged_analysis.box.get("ac|propulsion|num_engines"))
    assert float(results["engines_weight"]) == pytest.approx(float(results["engine_weight"]) * count, rel=1e-6)


# =============================================================================================
# Performance identities
# =============================================================================================


@pytest.mark.verification
@pytest.mark.slow
def test_the_takeoff_speeds_are_ordered(results):
    """The takeoff safety speed must exceed the decision speed.

    V\\ :sub:`2` is reached after rotation, which happens at or after V\\ :sub:`1`. A model that
    reported them the other way round would be describing a takeoff that cannot be flown.
    """
    assert float(results["V2"]) > float(results["V1"]) > 0.0


@pytest.mark.verification
@pytest.mark.slow
def test_the_field_length_is_balanced_and_positive(results):
    """Continue and abort distances equal, and both physically sensible for a narrow-body."""
    field, abort = float(results["takeoff_field_length"]), float(results["abort_distance"])
    assert field == pytest.approx(abort, rel=1e-6)
    assert 3000.0 < field < 12000.0, f"{field} ft is not a credible balanced field length"


@pytest.mark.verification
@pytest.mark.slow
def test_the_reserve_mission_costs_fuel_and_the_block_mission_costs_more(results):
    """Fuel accumulates monotonically across the mission, so the ordering is fixed."""
    block, total = float(results["block_fuel"]), float(results["total_fuel"])
    assert 0.0 < block < total


@pytest.mark.verification
@pytest.mark.slow
def test_flaps_increase_the_maximum_lift_coefficient(results):
    """Takeoff CLmax must exceed clean CLmax, or the high-lift model has the wrong sign."""
    assert float(results["CLmax_takeoff"]) > float(results["CLmax_cruise"]) > 0.0


@pytest.mark.verification
@pytest.mark.slow
def test_the_tail_surfaces_are_a_credible_fraction_of_the_wing(results, converged_analysis):
    """Tail areas are sized by the box; they must still come out the size tails are."""
    wing = float(converged_analysis.box.get("ac|geom|wing|S_ref", units="m**2"))
    assert 0.10 < float(results["hstab_area"]) / wing < 0.40
    assert 0.08 < float(results["vstab_area"]) / wing < 0.30


@pytest.mark.verification
@pytest.mark.slow
def test_every_phase_takes_a_positive_amount_of_time(results):
    """A solved phase duration that came out negative would still integrate, and be nonsense."""
    for phase in ("climb", "cruise", "descent", "loiter"):
        duration = float(results[f"{phase}_duration"])
        assert duration > 0.0, f"{phase} duration is {duration} min"


@pytest.mark.verification
@pytest.mark.slow
def test_the_throttle_histories_are_physically_bounded(results):
    """Throttle must lie between idle and the deck's rated setting in every phase.

    Not a certification requirement -- that is stated separately in the case file, with a
    source -- but a statement about what the reported numbers can mean at all.
    """
    for phase in ("climb", "cruise", "descent"):
        history = results[f"{phase}_throttle"]
        assert history.min() >= 0.0, f"{phase} throttle goes negative"
        assert history.max() <= 1.05, f"{phase} throttle exceeds the deck by more than 5%"
