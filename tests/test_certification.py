"""Unit and integration: requirements, their provenance, and the traceability matrix."""

from __future__ import annotations

import numpy as np
import pytest

from cdadt import (
    BalancedFieldLength,
    CertificationBasis,
    EngineOutClimbGradient,
    Requirement,
    RequirementError,
    RequirementSpec,
    ResponseCatalog,
    ResponseLimit,
    ThrottleLimit,
)

CATALOG = ResponseCatalog()


class FakeBox:
    """The smallest thing a requirement can be evaluated against.

    Requirements only ever ask a box for a value at a path, so a dictionary is enough. Using one
    keeps these tests about the requirement logic rather than about OpenConcept.
    """

    def __init__(self, values: dict[str, float | np.ndarray]) -> None:
        self._values = values

    def get(self, path: str, units: str | None = None):
        """Return the stored value, ignoring units -- the fake stores what it is asked for."""
        return self._values[path]

    def has(self, path: str) -> bool:
        """Return whether the fake holds that path."""
        return path in self._values


def _field_length(limit: float = 8000.0) -> BalancedFieldLength:
    """Return a balanced field length requirement with a stated source."""
    return BalancedFieldLength(
        limit=limit, regulation="14 CFR 25.113", source="8000 ft dry runway at sea level, ISA", units="ft"
    )


# =============================================================================================
# Provenance
# =============================================================================================


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["regulation", "source"])
def test_a_requirement_must_name_its_regulation_and_source(missing):
    """Both are required at construction, not only in the case file."""
    arguments = {"limit": 8000.0, "regulation": "14 CFR 25.113", "source": "a runway"}
    arguments[missing] = ""
    with pytest.raises(RequirementError, match="must"):
        BalancedFieldLength(**arguments)


@pytest.mark.unit
def test_a_requirement_refuses_options_it_does_not_understand():
    """A misspelled option would otherwise be silently dropped."""
    with pytest.raises(RequirementError, match="does not understand"):
        BalancedFieldLength(limit=8000.0, regulation="r", source="s", options={"phase": "climb"})


@pytest.mark.unit
def test_a_throttle_limit_must_say_which_phase():
    """Without it there is no response to read."""
    with pytest.raises(RequirementError, match="which 'phase'"):
        ThrottleLimit(limit=1.0, regulation="design", source="engine deck")


@pytest.mark.unit
def test_a_response_limit_must_say_what_it_limits():
    """The generic escape hatch still has to be specific."""
    with pytest.raises(RequirementError, match="which 'response'"):
        ResponseLimit(limit=1.0, regulation="design", source="a reason")


# =============================================================================================
# Construction from a case file
# =============================================================================================


@pytest.mark.unit
def test_every_shipped_requirement_registers_itself_under_its_type():
    """The case file names a requirement by ``type``; the registry is how that resolves."""
    for kind in ("balanced_field_length", "engine_out_climb_gradient", "throttle_limit", "response_limit"):
        assert kind in Requirement.registry


@pytest.mark.unit
def test_an_unknown_requirement_type_lists_the_ones_that_exist():
    """A typo in a case file should not require reading the source to fix."""
    spec = RequirementSpec("balanced_feild_length", 8000.0, "r", "s")
    with pytest.raises(RequirementError, match="Available:"):
        Requirement.from_spec(spec)


@pytest.mark.unit
def test_a_requirement_built_from_a_spec_keeps_its_provenance():
    """The regulation and source travel from the case file to the report unchanged."""
    spec = RequirementSpec("balanced_field_length", 8000.0, "14 CFR 25.113", "an 8000 ft runway", units="ft")
    requirement = Requirement.from_spec(spec)
    assert isinstance(requirement, BalancedFieldLength)
    assert requirement.regulation == "14 CFR 25.113"
    assert requirement.source == "an 8000 ft runway"


@pytest.mark.unit
def test_throttle_limits_in_different_phases_get_different_names():
    """Two requirements with one name would silently replace one another as constraints."""
    climb = ThrottleLimit(1.0, "design", "deck", options={"phase": "climb"})
    cruise = ThrottleLimit(1.0, "design", "deck", options={"phase": "cruise"})
    assert climb.name != cruise.name
    assert climb.response_name() == "climb_throttle"
    assert climb.title != cruise.title


@pytest.mark.unit
def test_a_basis_refuses_two_requirements_with_the_same_name():
    """OpenMDAO would keep only one of the two constraints."""
    with pytest.raises(RequirementError, match="named"):
        CertificationBasis([_field_length(), _field_length(7000.0)])


# =============================================================================================
# Evaluation
# =============================================================================================


@pytest.mark.unit
def test_an_upper_limit_is_met_below_it_and_violated_above_it():
    """Margin is positive on the satisfying side, whichever way the inequality runs."""
    requirement = _field_length()
    box = FakeBox({requirement.path(CATALOG): 6000.0})
    result = requirement.evaluate(box, CATALOG)
    assert result.satisfied and result.margin == pytest.approx(2000.0) and result.status == "MET"

    result = requirement.evaluate(FakeBox({requirement.path(CATALOG): 9000.0}), CATALOG)
    assert not result.satisfied and result.margin == pytest.approx(-1000.0) and result.status == "VIOLATED"


@pytest.mark.unit
def test_a_lower_limit_runs_the_other_way():
    """The engine-out gradient must not fall below the minimum."""
    requirement = EngineOutClimbGradient(0.024, "14 CFR 25.121(b)", "two-engine aeroplane", units="rad")
    met = requirement.evaluate(FakeBox({requirement.path(CATALOG): 0.05}), CATALOG)
    violated = requirement.evaluate(FakeBox({requirement.path(CATALOG): 0.01}), CATALOG)
    assert met.satisfied and met.margin == pytest.approx(0.026)
    assert not violated.satisfied and violated.margin == pytest.approx(-0.014)


@pytest.mark.unit
def test_a_requirement_sitting_on_its_limit_is_reported_as_active():
    """An active requirement is one that shaped the design, and that is worth distinguishing."""
    requirement = _field_length()
    result = requirement.evaluate(FakeBox({requirement.path(CATALOG): 8000.0}), CATALOG)
    assert result.satisfied and result.active and result.status == "ACTIVE"


@pytest.mark.unit
def test_a_vector_response_is_reduced_to_the_node_that_governs():
    """A throttle history is one requirement, not twenty-one, and one node decides it."""
    requirement = ThrottleLimit(1.0, "design", "deck", options={"phase": "climb"})
    box = FakeBox({requirement.path(CATALOG): np.array([0.5, 0.9, 1.2, 0.7])})
    result = requirement.evaluate(box, CATALOG)
    assert result.value == pytest.approx(1.2)
    assert not result.satisfied


@pytest.mark.unit
def test_an_empty_basis_says_so_rather_than_printing_an_empty_table():
    """An unconstrained study is a valid study; the report must not imply otherwise."""
    basis = CertificationBasis([])
    assert "No certification basis" in basis.traceability_matrix(FakeBox({}), CATALOG)


# =============================================================================================
# The traceability matrix
# =============================================================================================


@pytest.mark.integration
def test_the_traceability_matrix_names_every_regulation_and_source(built_box, optimization_config):
    """The artefact the module exists to produce, evaluated on a real converged design."""
    basis = CertificationBasis.from_specs(optimization_config.optimization.requirements)
    matrix = basis.traceability_matrix(built_box, CATALOG)

    for requirement in basis:
        assert requirement.regulation in matrix
        assert requirement.source in matrix
        assert requirement.title in matrix
    assert "requirements met" in matrix


@pytest.mark.integration
def test_the_shipped_basis_is_satisfied_by_the_baseline_aircraft(built_box, optimization_config):
    """The baseline must be feasible, or the optimization starts outside its own basis."""
    basis = CertificationBasis.from_specs(optimization_config.optimization.requirements)
    violated = [result.requirement.name for result in basis.evaluate(built_box, CATALOG) if not result.satisfied]
    assert not violated, f"The baseline B738 violates {violated}"
