"""Unit and integration: constraints, their provenance, and the traceability matrix."""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt import CertificationBasis, Constraint, ConstraintError, ConstraintSpec, ResponseCatalog, Scaling
from cdadt.config import Bounds

CATALOG = ResponseCatalog()


class FakeBox:
    """The smallest thing a constraint can be evaluated against.

    Constraints only ever ask a box for a value at a path, so a dictionary is enough. Using one
    keeps these tests about the constraint logic rather than about OpenConcept.
    """

    def __init__(self, values: dict[str, float | np.ndarray]) -> None:
        self._values = values

    def get(self, path: str, units: str | None = None):
        """Return the stored value, ignoring units -- the fake stores what it is asked for."""
        return self._values[path]

    def has(self, path: str) -> bool:
        """Return whether the fake holds that path."""
        return path in self._values


def _constraint(**overrides) -> Constraint:
    """Return a field-length constraint with stated provenance."""
    settings = {
        "name": "takeoff_field_length",
        "bounds": Bounds(upper=8000.0),
        "units": "ft",
        "regulation": "14 CFR 25.113",
        "source": "8000 ft dry runway at sea level, ISA",
    }
    settings.update(overrides)
    return Constraint(ConstraintSpec(**settings), CATALOG)


# =============================================================================================
# Naming and provenance
# =============================================================================================


@pytest.mark.unit
def test_a_constraint_resolves_a_response_name_to_a_path_and_units():
    """Response names are preferred over raw paths because they carry their own units."""
    constraint = _constraint(units=None)
    assert constraint.path == "mission.bfl.distance_continue"
    assert constraint.units == "ft"
    assert constraint.name == "takeoff_field_length"


@pytest.mark.unit
def test_a_raw_path_is_passed_through_unchanged():
    """Anything the catalogue does not know is the box's name, and the box gives the error."""
    constraint = Constraint(ConstraintSpec("mission.cruise.fltcond|M", Bounds(upper=0.82)), CATALOG)
    assert constraint.path == "mission.cruise.fltcond|M"
    assert constraint.units is None


@pytest.mark.unit
def test_units_stated_in_the_case_file_win_over_the_response_default():
    """A case file may state a limit in whatever unit the regulation is written in."""
    assert _constraint(units="m").units == "m"


@pytest.mark.unit
def test_provenance_is_optional_and_a_constraint_knows_whether_it_has_any():
    """Naming both a regulation and a source is what makes a row certification evidence."""
    assert _constraint().is_traceable
    assert not _constraint(regulation="", source="").is_traceable
    assert not _constraint(source="").is_traceable, "a regulation with no source is not traceable"
    assert not _constraint(regulation="").is_traceable


@pytest.mark.unit
def test_a_title_defaults_to_the_name_so_a_report_always_has_one():
    """And a stated title replaces it."""
    assert _constraint().title == "takeoff_field_length"
    assert _constraint(title="Balanced field length").title == "Balanced field length"


# =============================================================================================
# Evaluation
# =============================================================================================


@pytest.mark.unit
def test_an_upper_limit_is_met_below_it_and_violated_above_it():
    """Margin is positive on the satisfying side, whichever way the bound runs."""
    constraint = _constraint()
    met = constraint.evaluate(FakeBox({constraint.path: 6000.0}))
    assert met.satisfied and met.margin == pytest.approx(2000.0) and met.status == "MET"

    violated = constraint.evaluate(FakeBox({constraint.path: 9000.0}))
    assert not violated.satisfied and violated.margin == pytest.approx(-1000.0)
    assert violated.status == "VIOLATED"


@pytest.mark.unit
def test_a_lower_limit_runs_the_other_way():
    """The engine-out gradient must not fall below the minimum."""
    constraint = Constraint(ConstraintSpec("engine_out_climb_gradient", Bounds(lower=0.024), units="rad"), CATALOG)
    met = constraint.evaluate(FakeBox({constraint.path: 0.05}))
    violated = constraint.evaluate(FakeBox({constraint.path: 0.01}))
    assert met.satisfied and met.margin == pytest.approx(0.026)
    assert not violated.satisfied and violated.margin == pytest.approx(-0.014)


@pytest.mark.unit
def test_a_two_sided_band_is_only_as_compliant_as_its_nearest_side():
    """Written exactly as ``add_constraint("climb.throttle", lower=0.01, upper=1.05)``."""
    constraint = Constraint(ConstraintSpec("climb_throttle", Bounds(lower=0.01, upper=1.05)), CATALOG)

    # Node 0 has margin min(1.05-0.5, 0.5-0.01) = 0.49; node 1 has min(0.45, 0.59) = 0.45.
    # The governing node is the tighter one, so the band reports 0.45 at a value of 0.6.
    middle = constraint.evaluate(FakeBox({constraint.path: np.array([0.5, 0.6])}))
    assert middle.satisfied and middle.margin == pytest.approx(0.45)
    assert middle.value == pytest.approx(0.6)

    near_top = constraint.evaluate(FakeBox({constraint.path: np.array([0.5, 1.04])}))
    assert near_top.satisfied and near_top.margin == pytest.approx(0.01)

    below = constraint.evaluate(FakeBox({constraint.path: np.array([0.5, 0.005])}))
    assert not below.satisfied


@pytest.mark.unit
def test_an_equality_is_satisfied_only_at_the_value():
    """Its margin is zero when met and negative by the distance otherwise."""
    constraint = Constraint(ConstraintSpec("mission_range_flown", Bounds(equals=2800.0), units="nmi"), CATALOG)
    exact = constraint.evaluate(FakeBox({constraint.path: 2800.0}))
    off = constraint.evaluate(FakeBox({constraint.path: 2750.0}))
    assert exact.satisfied and exact.margin == pytest.approx(0.0)
    assert not off.satisfied and off.margin == pytest.approx(-50.0)


@pytest.mark.unit
def test_a_vector_response_is_reduced_to_the_node_that_governs():
    """A throttle history is one constraint, not one per node, and one node decides it."""
    constraint = Constraint(ConstraintSpec("climb_throttle", Bounds(upper=1.0)), CATALOG)
    result = constraint.evaluate(FakeBox({constraint.path: np.array([0.5, 0.9, 1.2, 0.7])}))
    assert result.value == pytest.approx(1.2)
    assert not result.satisfied


@pytest.mark.unit
def test_indices_narrow_a_vector_constraint_to_the_nodes_that_matter():
    """``add_constraint(..., indices=[0], upper=1.0)`` appears in OpenConcept's own examples."""
    constraint = Constraint(ConstraintSpec("climb_throttle", Bounds(upper=1.0), indices=[0]), CATALOG)
    result = constraint.evaluate(FakeBox({constraint.path: np.array([0.5, 1.2])}))
    assert result.value == pytest.approx(0.5)
    assert result.satisfied, "the offending node was excluded on purpose"


@pytest.mark.unit
def test_a_constraint_sitting_on_its_bound_is_reported_as_active():
    """An active constraint is one that shaped the design, and that is worth distinguishing."""
    constraint = _constraint()
    result = constraint.evaluate(FakeBox({constraint.path: 8000.0}))
    assert result.satisfied and result.active and result.status == "ACTIVE"


@pytest.mark.unit
def test_constraints_and_results_repr_as_what_they_assert():
    """Both end up in debugger frames while a certification argument is being checked."""
    constraint = _constraint()
    assert repr(constraint) == "Constraint('takeoff_field_length', <= 8000)"
    result = constraint.evaluate(FakeBox({constraint.path: 6000.0}))
    assert repr(result) == "ConstraintResult('takeoff_field_length', 6000.0000, MET)"


# =============================================================================================
# The basis
# =============================================================================================


@pytest.mark.unit
def test_a_basis_refuses_two_constraints_with_the_same_name():
    """OpenMDAO would keep only one of the two."""
    with pytest.raises(ConstraintError, match="named"):
        CertificationBasis([_constraint(), _constraint(bounds=Bounds(upper=7000.0))])


@pytest.mark.unit
def test_an_empty_basis_says_so_rather_than_printing_an_empty_table():
    """An unconstrained study is a valid study; the report must not imply otherwise."""
    basis = CertificationBasis([])
    assert len(basis) == 0
    assert "No constraints were declared" in basis.traceability_matrix(FakeBox({}))


@pytest.mark.unit
def test_a_basis_separates_the_traceable_constraints_from_the_rest():
    """The distinction the whole module exists to make."""
    plain = Constraint(ConstraintSpec("climb_throttle", Bounds(lower=0.01, upper=1.05)), CATALOG)
    basis = CertificationBasis([_constraint(), plain])
    assert len(basis) == 2
    assert [c.name for c in basis.traceable] == ["takeoff_field_length"]
    assert [c.name for c in basis] == ["takeoff_field_length", "climb_throttle"]
    assert repr(basis) == "CertificationBasis(2 constraints)"


@pytest.mark.unit
def test_the_matrix_reports_provenance_where_there_is_some_and_says_so_where_there_is_not():
    """A report that implied every row was certification evidence would be wrong."""
    plain = Constraint(ConstraintSpec("climb_throttle", Bounds(lower=0.01, upper=1.05)), CATALOG)
    basis = CertificationBasis([_constraint(), plain])
    box = FakeBox({"mission.bfl.distance_continue": 6000.0, "mission.climb.throttle": np.array([0.5])})

    matrix = basis.traceability_matrix(box)
    assert "14 CFR 25.113" in matrix
    assert "8000 ft dry runway at sea level, ISA" in matrix
    assert "Design constraints with no stated regulation or source: climb_throttle" in matrix
    assert "2 of 2 constraints met" in matrix


@pytest.mark.unit
def test_a_basis_of_only_plain_constraints_prints_no_provenance_section():
    """There is nothing to trace, and an empty heading would suggest there should be."""
    plain = Constraint(ConstraintSpec("climb_throttle", Bounds(upper=1.05)), CATALOG)
    matrix = CertificationBasis([plain]).traceability_matrix(FakeBox({"mission.climb.throttle": np.array([0.5])}))
    assert "Where each limit came from" not in matrix
    assert "no stated regulation or source" in matrix


@pytest.mark.unit
def test_a_basis_of_only_traceable_constraints_prints_no_design_note():
    """Symmetrically: nothing untraceable, so no note about untraceable things."""
    matrix = CertificationBasis([_constraint()]).traceability_matrix(FakeBox({"mission.bfl.distance_continue": 6000.0}))
    assert "Where each limit came from" in matrix
    assert "no stated regulation or source" not in matrix


@pytest.mark.unit
def test_a_constraint_scales_itself_by_its_bound_unless_told_otherwise():
    """A gradient in hundredths of a radian is invisible beside a field length in thousands."""
    assert _constraint().spec.bounds.magnitude == 8000.0
    assert _constraint(scaling=Scaling(ref=1.0)).spec.scaling.ref == 1.0


@pytest.mark.unit
def test_registering_a_constraint_declares_it_on_the_group_as_stated():
    """What the driver is actually given, read back off a real OpenMDAO group.

    Declared on a bare :class:`openmdao.api.Group` rather than a recording double, because the
    claim is about what OpenMDAO accepts and stores -- a double would only confirm that
    ``register`` calls a method. Before setup a group keeps these in ``_static_responses``, which
    is private but is the dependency's own storage; asserting against it is the honest reading.

    Note the bound arrives *scaled*: 8000 ft against a ref of 8000 is 1.0. That is the point of
    the default scaling, and it is what would be silently lost if ``register`` stopped passing it.
    """
    group = om.Group()
    _constraint().register(group)

    recorded = group._static_responses["takeoff_field_length"]
    assert recorded["units"] == "ft"
    assert recorded["upper"] == pytest.approx(1.0)
    assert recorded["ref"] == pytest.approx(8000.0)
    assert recorded.get("indices") is None, "a scalar constraint must not be given indices"


@pytest.mark.unit
def test_a_constraint_may_be_applied_to_chosen_nodes_of_a_vector_response():
    """Throttle is constrained over the whole climb; a case file may name a subset instead.

    Without this the ``indices`` key would parse, be reported, and never reach the driver -- so
    the constraint would silently apply to every node.
    """
    group = om.Group()
    _constraint(name="climb_throttle", bounds=Bounds(upper=1.05), units=None, indices=[0, 2]).register(group)

    # OpenMDAO wraps them in an indexer of its own, so read the array back out of it.
    assert group._static_responses["climb_throttle"]["indices"].as_array().tolist() == [0, 2]


# =============================================================================================
# Against a real run
# =============================================================================================


@pytest.mark.integration
def test_the_traceability_matrix_names_every_regulation_and_source(built_box, optimization_config):
    """The artefact the module exists to produce, evaluated on a real converged design."""
    basis = CertificationBasis.from_specs(optimization_config.constraints, CATALOG)
    matrix = basis.traceability_matrix(built_box)

    for constraint in basis.traceable:
        assert constraint.regulation in matrix
        assert constraint.source in matrix
    for constraint in basis:
        assert constraint.title in matrix
    assert "constraints met" in matrix


@pytest.mark.integration
def test_the_shipped_basis_is_satisfied_by_the_baseline_aircraft(built_box, optimization_config):
    """The baseline must be feasible, or the optimization starts outside its own basis."""
    basis = CertificationBasis.from_specs(optimization_config.constraints, CATALOG)
    violated = [result.constraint.name for result in basis.evaluate(built_box) if not result.satisfied]
    assert not violated, f"The baseline B738 violates {violated}"
