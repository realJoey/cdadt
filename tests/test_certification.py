"""Tests for the certification basis: margins, status, registration and the matrix."""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt import (
    ApproachSpeed,
    BalancedFieldLength,
    CertificationBasis,
    EngineOutClimbGradient,
    RequirementResult,
    Sense,
    ThrottleMargin,
)

pytestmark = pytest.mark.unit


class _StubProblem:
    """A stand-in that returns a prescribed value for a prescribed path."""

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get_val(self, path: str, units: str | None = None) -> np.ndarray:
        """Return the stubbed value; the units are asserted by the caller's expectations."""
        return np.atleast_1d(np.asarray(self._values[path], dtype=float))


def test_an_upper_limit_margin_is_limit_minus_value():
    """A field length shorter than the runway has positive margin."""
    requirement = BalancedFieldLength(limit=8000.0, source="test runway")
    result = requirement.evaluate(_StubProblem({requirement.path: 6000.0}))
    assert requirement.sense == Sense.UPPER
    assert result.value == pytest.approx(6000.0)
    assert result.margin == pytest.approx(2000.0)
    assert result.satisfied
    assert result.status == "MET"


def test_an_upper_limit_reports_the_worst_element_of_a_vector():
    """Throttle must be within limit everywhere, so the peak is what is reported."""
    requirement = ThrottleMargin(phase="climb", limit=1.0, source="engine deck")
    result = requirement.evaluate(_StubProblem({requirement.path: [0.8, 1.05, 0.9]}))
    assert result.value == pytest.approx(1.05)
    assert not result.satisfied
    assert result.status == "NOT MET"


def test_a_lower_limit_reports_the_worst_element_and_the_right_sign():
    """A climb gradient must hold everywhere, so the minimum is what is reported."""
    requirement = EngineOutClimbGradient(limit=0.024, source="two-engine")
    result = requirement.evaluate(_StubProblem({requirement.path: [0.05, 0.03]}))
    assert requirement.sense == Sense.LOWER
    assert result.value == pytest.approx(0.03)
    assert result.margin == pytest.approx(0.03 - np.arctan(0.024))


def test_the_climb_gradient_limit_is_converted_from_gradient_to_angle():
    """The regulation states a tangent; OpenConcept reports an angle. They are not the same."""
    requirement = EngineOutClimbGradient(limit=0.024, source="two-engine")
    assert requirement.gradient == pytest.approx(0.024)
    assert requirement.limit == pytest.approx(np.arctan(0.024))
    assert requirement.limit < requirement.gradient  # arctan(x) < x for x > 0


def test_a_binding_requirement_reads_as_active_not_violated():
    """A converged optimizer lands a hair either side of the bound."""
    requirement = BalancedFieldLength(limit=8000.0, source="test runway")
    result = requirement.evaluate(_StubProblem({requirement.path: 8000.0 + 1e-9}))
    assert result.status == "ACTIVE"
    assert result.active


def test_relative_margin_makes_feet_and_radians_comparable():
    """Sorting the matrix by absolute margin would always rank the field length last."""
    field = BalancedFieldLength(limit=8000.0, source="s").evaluate(
        _StubProblem({"mission.bfl.distance_continue": 7600.0})
    )
    climb = EngineOutClimbGradient(limit=0.024, source="s").evaluate(
        _StubProblem({"mission.engineoutclimb.gamma": 0.0243})
    )
    assert field.margin > climb.margin  # 400 ft against 0.0003 rad
    assert climb.relative_margin < field.relative_margin  # but the climb is the binding one


def test_duplicate_requirement_names_are_rejected():
    """Two rows that cannot be told apart make the matrix useless."""
    with pytest.raises(ValueError, match="duplicated"):
        CertificationBasis(
            [
                ThrottleMargin(phase="climb", limit=1.0, source="a"),
                ThrottleMargin(phase="climb", limit=1.05, source="b"),
            ]
        )


class _ConstraintRecorder:
    """Captures what a requirement asks the model to constrain.

    OpenMDAO stores constraints added before setup where they cannot be read back, so the
    contract under test -- what ``register`` asks for -- is captured directly.
    """

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def add_constraint(self, name: str, **kwargs) -> None:
        """Record the constraint request."""
        self.calls[name] = kwargs


def test_constraints_are_registered_scaled_by_their_limit():
    """A field length in thousands of feet and a gradient in hundredths must weigh the same."""
    recorder = _ConstraintRecorder()
    CertificationBasis(
        [
            BalancedFieldLength(limit=8000.0, source="s"),
            EngineOutClimbGradient(limit=0.024, source="s"),
        ]
    ).register(recorder)

    field = recorder.calls["mission.bfl.distance_continue"]
    climb = recorder.calls["mission.engineoutclimb.gamma"]
    assert field["upper"] == pytest.approx(8000.0)
    assert field["units"] == "ft"
    assert field["scaler"] == pytest.approx(1.0 / 8000.0)
    assert climb["lower"] == pytest.approx(np.arctan(0.024))
    assert climb["units"] == "rad"
    assert climb["scaler"] == pytest.approx(1.0 / np.arctan(0.024))
    # Both constraints are order one once scaled, which is the whole point.
    assert field["upper"] * field["scaler"] == pytest.approx(1.0)
    assert climb["lower"] * climb["scaler"] == pytest.approx(1.0)


def test_a_dimensionless_requirement_registers_without_units():
    """Throttle has no units, and passing 'None' as a string would be a unit named None."""
    recorder = _ConstraintRecorder()
    ThrottleMargin(phase="climb", limit=1.0, source="s").register(recorder)
    assert recorder.calls["mission.climb.throttle"]["units"] is None


def test_the_traceability_matrix_names_every_limit_source():
    """A limit without provenance is indistinguishable from a guess."""
    basis = CertificationBasis(
        [
            BalancedFieldLength(limit=8000.0, source="8000 ft dry runway, sea level ISA"),
            ThrottleMargin(phase="climb", limit=1.0, source="engine deck rated condition"),
        ]
    )
    matrix = basis.traceability_matrix(
        _StubProblem({"mission.bfl.distance_continue": 6000.0, "mission.climb.throttle": [0.9]})
    )
    assert "14 CFR 25.113" in matrix
    assert "8000 ft dry runway, sea level ISA" in matrix
    assert "engine deck rated condition" in matrix
    assert "2 of 2 requirements met" in matrix


@pytest.mark.integration
def test_the_approach_speed_requirement_builds_and_computes_vref():
    """The requirement contributes the landing analysis the mission does not fly."""
    requirement = ApproachSpeed(limit=140.0, source="category C")
    model = om.Group()
    parameters = model.add_subsystem("parameters", om.IndepVarComp(), promotes_outputs=["*"])
    parameters.add_output("ac|aero|landing_flap_deg", 40.0, units="deg")
    parameters.add_output("ac|aero|CLmax_cruise", 1.4274)
    parameters.add_output("ac|geom|wing|c4sweep", 25.0, units="deg")
    parameters.add_output("ac|geom|wing|toverc", 0.12)
    parameters.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
    parameters.add_output("ac|weights|MLW", 62676.0, units="kg")
    requirement.build(model)

    problem = om.Problem(model=model, reports=False)
    problem.setup(check=False)
    problem.run_model()

    clmax_land = problem.get_val("approach_speed.CLmax_land").item()
    vstall = problem.get_val("approach_speed.Vstall_eas", units="m/s").item()
    vref = problem.get_val(requirement.path, units="m/s").item()

    assert clmax_land > 1.4274  # landing flaps add lift to the clean maximum
    assert vstall == pytest.approx(np.sqrt(2 * 62676.0 * 9.80665 / 1.225 / 124.6 / clmax_land), rel=1e-6)
    assert vref == pytest.approx(1.23 * vstall, rel=1e-9)


def test_requirement_result_active_tolerance_is_relative():
    """A tolerance in absolute units would mean something different for feet and radians."""
    tight = RequirementResult(
        requirement=BalancedFieldLength(limit=8000.0, source="s"),
        value=8000.0,
        limit=8000.0,
        units="ft",
        margin=1e-3,
    )
    assert tight.active  # 1e-3 ft against an 8000 ft limit is binding
    loose = RequirementResult(
        requirement=EngineOutClimbGradient(limit=0.024, source="s"),
        value=0.03,
        limit=0.0239,
        units="rad",
        margin=1e-3,
    )
    assert not loose.active  # 1e-3 rad against a 0.024 rad limit is a real margin
