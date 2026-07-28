"""Validation: does the converged aircraft resemble a real Boeing 737-800?

:mod:`tests.test_validation` establishes that cdadt reproduces OpenConcept exactly. That is
verification of the coupling, and it would pass just as well if OpenConcept's physics were
nonsense -- both sides would be wrong together.

This module asks the other question. It takes the converged design and checks it against things
known independently of the model: published figures for the aeroplane it is meant to represent,
and dimensionless groups whose plausible ranges for a transport jet come from outside any of
this code.

Two honest caveats, stated here rather than only in the documentation.

**The published figures are approximate.** They are widely quoted specification values for the
737-800, not measurements traceable to a certification document, and the tolerances below are
correspondingly generous. A check that passes at plus or minus ten per cent is a plausibility
check, not a certification-grade validation.

**Some checks are consistency, not validation.** Wing span is computed from reference area and
aspect ratio, both of which are *inputs* taken from the same source as the published span. That
it agrees is a check that the geometry is assembled correctly -- worth having, and marked as
such -- not independent evidence that the model is right.
"""

from __future__ import annotations

import math

import pytest

#: Standard gravity, for weight-to-force conversions.
G = 9.80665

#: Publicly quoted Boeing 737-800 figures, with the tolerance each is checked to. These are
#: specification values from general reference sources, not certification data; the tolerances
#: reflect that. Winglets, engine variant and operator configuration all move the real numbers.
PUBLISHED = {
    "MTOW": (79_010.0, "kg", 0.10, "Quoted maximum takeoff weight, 737-800"),
    "OEW": (41_410.0, "kg", 0.10, "Quoted operating empty weight, 737-800, typical configuration"),
}

#: Dimensionless groups and their plausible ranges for a narrow-body transport jet. The bounds
#: come from transport-aircraft design practice, not from this model or from OpenConcept.
BANDS = {
    "cruise Mach": (0.72, 0.85),
    "cruise lift-to-drag ratio": (14.0, 20.0),
    "cruise TSFC, lb/lbf/hr": (0.50, 0.75),
    "wing loading, kg/m2": (500.0, 750.0),
    "sea-level thrust-to-weight": (0.25, 0.40),
    "operating empty weight fraction": (0.45, 0.60),
    "fuel fraction with reserves": (0.15, 0.35),
}


@pytest.fixture(scope="module")
def design(converged_analysis):
    """Return the converged aircraft's headline results and derived cruise quantities."""
    box = converged_analysis.box
    results = converged_analysis.results()
    middle = box.num_nodes // 2  # mid-cruise, away from either end of the phase

    weight = float(box.get("mission.cruise.weight", units="kg")[middle])
    drag = float(box.get("mission.cruise.drag", units="N")[middle])
    thrust = float(box.get("mission.cruise.thrust", units="N")[middle])
    fuel_flow = float(box.get("mission.cruise.propulsion_multiplier.fuel_flow", units="kg/s")[middle])
    area = float(box.get("ac|geom|wing|S_ref", units="m**2"))
    aspect_ratio = float(box.get("ac|geom|wing|AR"))
    installed_thrust = float(box.get("ac|propulsion|engine|rating", units="N")) * float(
        box.get("ac|propulsion|num_engines")
    )

    return {
        "results": results,
        "cruise Mach": float(box.get("mission.cruise.fltcond|M")[middle]),
        "cruise lift coefficient": float(box.get("mission.cruise.fltcond|CL")[middle]),
        "cruise lift-to-drag ratio": weight * G / drag,
        "thrust over drag": thrust / drag,
        # Mass-specific fuel consumption in kg/(kgf*hr) is numerically identical to lb/(lbf*hr).
        "cruise TSFC, lb/lbf/hr": fuel_flow / (thrust / G) * 3600.0,
        "wing loading, kg/m2": float(results["MTOW"]) / area,
        "sea-level thrust-to-weight": installed_thrust / (float(results["MTOW"]) * G),
        "operating empty weight fraction": float(results["OEW"]) / float(results["MTOW"]),
        "fuel fraction with reserves": float(results["total_fuel"]) / float(results["MTOW"]),
        "span, m": math.sqrt(aspect_ratio * area),
    }


@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("quantity", sorted(PUBLISHED))
def test_the_sized_aircraft_matches_published_737_800_figures(design, quantity):
    """Weights must land near the real aeroplane's, within the stated tolerance.

    This is the check that the empirical weight buildup inside the black box is producing a
    737-sized aircraft rather than merely a self-consistent one. It is the strongest evidence
    available here that the physics is right and not just reproducible.
    """
    expected, units, tolerance, description = PUBLISHED[quantity]
    computed = float(design["results"][quantity])
    error = abs(computed - expected) / expected
    assert error <= tolerance, (
        f"{quantity}: cdadt {computed:.1f} {units} vs published {expected:.1f} {units} "
        f"({error:.1%} > {tolerance:.0%}). Reference: {description}."
    )


@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("quantity", sorted(BANDS))
def test_the_dimensionless_groups_are_those_of_a_transport_jet(design, quantity):
    """Each group must fall in the range transport-aircraft design practice expects.

    These bounds come from outside this repository and outside OpenConcept. A model can be
    internally consistent, reproduce its reference exactly, and still be describing an aeroplane
    that could not exist; this is what would catch that.
    """
    low, high = BANDS[quantity]
    value = design[quantity]
    assert low <= value <= high, f"{quantity} = {value:.4f}, outside the expected range [{low}, {high}]"


@pytest.mark.validation
@pytest.mark.slow
def test_cruise_is_actually_steady_level_flight(design):
    """Thrust must equal drag in cruise, to solver tolerance.

    Not a band or a plausibility check but an identity: the phase is defined by zero
    acceleration at zero vertical speed. If it does not hold, the trajectory being integrated is
    not the one the mission claims to fly, and every fuel number downstream is wrong.
    """
    assert design["thrust over drag"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.validation
@pytest.mark.slow
def test_the_weight_fractions_account_for_the_whole_aircraft(design):
    """Empty, payload and fuel fractions must sum to one, exactly.

    Independent of the closure test in :mod:`tests.test_validation`, which checks the same
    identity in kilograms. Expressed as fractions it is the form a design report quotes, and a
    rounding or unit error that survived one form would not survive both.
    """
    results = design["results"]
    total = float(results["MTOW"])
    fractions = float(results["OEW"]) / total + float(results["payload"]) / total + float(results["total_fuel"]) / total
    assert fractions == pytest.approx(1.0, rel=1e-9)


@pytest.mark.validation
@pytest.mark.slow
def test_the_geometry_is_assembled_consistently(design):
    """Span from area and aspect ratio must match the real aeroplane's.

    Marked as a consistency check rather than validation, and it matters which it is: reference
    area and aspect ratio are *inputs*, taken from the same source as the published span, so
    agreement confirms the geometry is assembled and converted correctly. It is not independent
    evidence about the physics.
    """
    published_span = 34.32  # metres, 737-800 without winglets
    assert design["span, m"] == pytest.approx(published_span, rel=0.02)


@pytest.mark.validation
@pytest.mark.slow
def test_the_design_carries_a_credible_reserve(design):
    """Reserve fuel must be a sensible fraction of block fuel for a Part 25 diversion.

    A 200 nmi diversion plus a 30-minute loiter should cost something like a sixth of a 2800 nmi
    block, and certainly neither a rounding error nor half the mission.
    """
    results = design["results"]
    reserve = float(results["total_fuel"]) - float(results["block_fuel"])
    assert 0.05 < reserve / float(results["block_fuel"]) < 0.35
