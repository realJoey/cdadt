"""Tests for the landing performance components.

Claim classes: ``derivative`` for the partials, ``validation`` for the hand calculations,
``unit`` for behavior.

These are the only components in cdadt that carry physics cdadt is responsible for, rather
than wrapping something already validated. They are therefore held to a higher standard than
the providers: every partial derivative is checked against complex step, and every formula
is checked against a hand calculation worked out in the test docstring.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials

from cdadt.certification.components.landing import ApproachSpeed, LandingFieldLength

NUM_NODES = 3


def _run(component, inputs):
    """Build one component, set inputs, run, and return the problem."""
    prob = om.Problem()
    prob.model.add_subsystem("comp", component, promotes=["*"])
    prob.setup(check=False, force_alloc_complex=True)
    for name, value in inputs.items():
        prob.set_val(name, value)
    prob.run_model()
    return prob


# ==============================================================================
# ApproachSpeed
# ==============================================================================
@pytest.mark.validation
def test_approach_speed_is_the_factor_times_the_stall_speed():
    """V_REF = k V_SR0, checked by hand.

    At V_SR0 = 60 m/s with the 14 CFR 25.125(a)(2) floor of k = 1.23, V_REF = 73.8 m/s.
    """
    prob = _run(
        ApproachSpeed(num_nodes=NUM_NODES),
        {
            "Vstall_land": np.full(NUM_NODES, 60.0),
            "approach_speed_factor": np.full(NUM_NODES, 1.23),
        },
    )
    assert prob.get_val("V_ref", units="m/s") == pytest.approx(np.full(NUM_NODES, 73.8))


@pytest.mark.derivative
def test_approach_speed_partials_match_complex_step():
    """Analytic partials agree with complex step.

    An optimizer consumes these derivatives. If they disagree with the function, the
    optimizer converges to a point that is optimal for a function nobody evaluated.
    """
    prob = _run(
        ApproachSpeed(num_nodes=NUM_NODES),
        {
            "Vstall_land": np.linspace(55.0, 65.0, NUM_NODES),
            "approach_speed_factor": np.linspace(1.23, 1.30, NUM_NODES),
        },
    )
    assert_check_partials(prob.check_partials(method="cs", out_stream=None), atol=1e-12, rtol=1e-12)


@pytest.mark.unit
def test_approach_speed_scales_with_both_inputs():
    """Raising either the stall speed or the factor raises the approach speed."""
    base = _run(
        ApproachSpeed(),
        {"Vstall_land": np.array([60.0]), "approach_speed_factor": np.array([1.23])},
    ).get_val("V_ref")
    faster_stall = _run(
        ApproachSpeed(),
        {"Vstall_land": np.array([66.0]), "approach_speed_factor": np.array([1.23])},
    ).get_val("V_ref")
    higher_factor = _run(
        ApproachSpeed(),
        {"Vstall_land": np.array([60.0]), "approach_speed_factor": np.array([1.30])},
    ).get_val("V_ref")

    assert faster_stall > base
    assert higher_factor > base


# ==============================================================================
# LandingFieldLength
# ==============================================================================
def _field_length_inputs(num_nodes=NUM_NODES):
    """Return a set of landing inputs with round numbers, for hand checking."""
    return {
        "V_ref": np.full(num_nodes, 70.0),
        "airborne_distance": np.full(num_nodes, 300.0),
        "mean_deceleration": np.full(num_nodes, 3.5),
        "dispatch_factor": np.full(num_nodes, 0.6),
    }


@pytest.mark.validation
def test_landing_distance_is_the_airborne_segment_plus_the_energy_ground_roll():
    """The demonstrated landing distance, checked by hand.

    Ground roll = V_ref^2 / (2 a) = 70^2 / (2 x 3.5) = 4900 / 7 = 700 m.
    Landing distance = 300 + 700 = 1000 m.
    """
    prob = _run(LandingFieldLength(num_nodes=NUM_NODES), _field_length_inputs())
    assert prob.get_val("landing_distance", units="m") == pytest.approx(np.full(NUM_NODES, 1000.0))


@pytest.mark.validation
def test_field_length_applies_the_dispatch_factor():
    """The field length required, checked by hand.

    14 CFR 121.195(b) allows dispatch only where the aircraft lands within 60% of the
    effective runway length, so a 1000 m demonstrated distance needs 1000 / 0.6 = 1666.67 m
    of runway.
    """
    prob = _run(LandingFieldLength(num_nodes=NUM_NODES), _field_length_inputs())
    assert prob.get_val("landing_field_length", units="m") == pytest.approx(np.full(NUM_NODES, 1000.0 / 0.6))


@pytest.mark.validation
def test_the_dispatch_factor_lengthens_rather_than_shortens_the_requirement():
    """A dispatch factor below one requires *more* runway than the distance demonstrated.

    Dividing where one should multiply is the classic error in this calculation, and it
    produces a shorter, entirely plausible-looking number. Checking the direction catches
    it where checking the magnitude alone would not.
    """
    prob = _run(LandingFieldLength(), {k: v[:1] for k, v in _field_length_inputs().items()})
    assert prob.get_val("landing_field_length", units="m").item() > prob.get_val("landing_distance", units="m").item()


@pytest.mark.derivative
def test_field_length_partials_match_complex_step():
    """Analytic partials of both outputs agree with complex step, at a non-degenerate point.

    Inputs vary across the nodes so that a partial which is only correct for equal inputs
    would fail here.
    """
    prob = _run(
        LandingFieldLength(num_nodes=NUM_NODES),
        {
            "V_ref": np.linspace(65.0, 78.0, NUM_NODES),
            "airborne_distance": np.linspace(280.0, 320.0, NUM_NODES),
            "mean_deceleration": np.linspace(2.8, 3.6, NUM_NODES),
            "dispatch_factor": np.linspace(0.55, 0.65, NUM_NODES),
        },
    )
    assert_check_partials(prob.check_partials(method="cs", out_stream=None), atol=1e-10, rtol=1e-10)


@pytest.mark.unit
def test_field_length_grows_with_the_square_of_approach_speed():
    """Doubling the approach speed quadruples the ground roll.

    The energy method's defining property. A component that had drifted to a linear form
    would still produce plausible distances at one speed.
    """
    inputs = {k: v[:1] for k, v in _field_length_inputs().items()}
    inputs["airborne_distance"] = np.array([0.0])

    slow = _run(LandingFieldLength(), inputs).get_val("landing_distance", units="m").item()
    inputs["V_ref"] = inputs["V_ref"] * 2.0
    fast = _run(LandingFieldLength(), inputs).get_val("landing_distance", units="m").item()

    assert fast == pytest.approx(4.0 * slow)


@pytest.mark.unit
def test_harder_braking_shortens_the_landing():
    """A higher mean deceleration gives a shorter ground roll."""
    inputs = {k: v[:1] for k, v in _field_length_inputs().items()}
    gentle = _run(LandingFieldLength(), inputs).get_val("landing_distance", units="m").item()
    inputs["mean_deceleration"] = np.array([5.0])
    hard = _run(LandingFieldLength(), inputs).get_val("landing_distance", units="m").item()
    assert hard < gentle
