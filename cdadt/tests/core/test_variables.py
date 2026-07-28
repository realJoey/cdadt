"""Tests for :mod:`cdadt.core.variables`.

Claim class: ``unit``. These prove that variable declarations validate their units and that
set algebra over them refuses to silently resolve a units disagreement. They prove nothing
about any physical model.

Unit validation runs against OpenMDAO's real unit library, not a stand-in list, because the
property being tested is agreement with the unit system the model will actually use.
"""

from __future__ import annotations

import pytest

from cdadt.core.variables import Variable, VariableConflictError, VariableSet

pytestmark = pytest.mark.unit


# ==============================================================================
# Variable
# ==============================================================================
@pytest.mark.parametrize(
    "units",
    ["m**2", "kg", "lbf", "kn", "ft/min", "N*m**-2", "deg", "s", "kg/s", "psi"],
    ids=lambda u: u,
)
def test_variable_accepts_units_openmdao_recognizes(units):
    """Every unit string cdadt uses in its variable declarations is valid in OpenMDAO."""
    variable = Variable("test|var", units)
    assert variable.units == units


def test_variable_accepts_none_for_dimensionless():
    """A dimensionless variable declares units of None rather than an empty string."""
    assert Variable("fltcond|CL", None).units is None


@pytest.mark.parametrize("units", ["furlongs", "m**", "kg/", "not_a_unit"])
def test_variable_rejects_invalid_units(units):
    """An unrecognized unit string is rejected at declaration, naming the variable."""
    with pytest.raises(ValueError, match="is not a valid OpenMDAO unit string"):
        Variable("ac|geom|wing|S_ref", units)


@pytest.mark.parametrize("name", ["", "   "])
def test_variable_rejects_empty_name(name):
    """A variable must have a non-empty name."""
    with pytest.raises(ValueError, match="non-empty string"):
        Variable(name, "m")


def test_variable_is_immutable():
    """Declarations are values: assigning to a field raises."""
    variable = Variable("ac|weights|MTOW", "kg")
    with pytest.raises(Exception):
        variable.units = "lbm"
    assert variable.units == "kg"


def test_variable_is_hashable_and_compares_by_content():
    """Two identical declarations are equal and hash alike, so sets deduplicate them."""
    a = Variable("ac|weights|MTOW", "kg", description="max takeoff mass")
    b = Variable("ac|weights|MTOW", "kg", description="max takeoff mass")
    assert a == b
    assert len({a, b}) == 1


def test_conflicts_with_detects_units_disagreement():
    """Same name, different units is a conflict."""
    assert Variable("ac|weights|MTOW", "kg").conflicts_with(Variable("ac|weights|MTOW", "lbm"))


def test_conflicts_with_detects_vectorization_disagreement():
    """Same name and units but different vectorization is a conflict."""
    scalar = Variable("throttle", None, vectorized=False)
    vector = Variable("throttle", None, vectorized=True)
    assert scalar.conflicts_with(vector)


def test_conflicts_with_ignores_description():
    """Descriptions may differ between declarations of the same variable."""
    a = Variable("drag", "N", vectorized=True, description="total drag")
    b = Variable("drag", "N", vectorized=True, description="airplane-axis drag")
    assert not a.conflicts_with(b)


def test_conflicts_with_is_false_for_different_names():
    """Different names never conflict, whatever their units."""
    assert not Variable("drag", "N").conflicts_with(Variable("thrust", "lbf"))


# ==============================================================================
# VariableSet
# ==============================================================================
@pytest.fixture
def aero_outputs():
    """Return a small set standing in for an aerodynamics discipline's outputs."""
    return VariableSet(
        [
            Variable("drag", "N", vectorized=True),
            Variable("ac|aero|CLmax_TO", None),
        ]
    )


def test_set_reports_length_and_names(aero_outputs):
    """A set knows how many declarations it holds and what they are called."""
    assert len(aero_outputs) == 2
    assert aero_outputs.names == frozenset({"drag", "ac|aero|CLmax_TO"})


def test_set_membership_by_name_and_by_declaration(aero_outputs):
    """Membership accepts a name, and accepts a declaration only if it matches exactly."""
    assert "drag" in aero_outputs
    assert Variable("drag", "N", vectorized=True) in aero_outputs
    # Same name, wrong units: not present.
    assert Variable("drag", "lbf", vectorized=True) not in aero_outputs
    assert "lift" not in aero_outputs


def test_set_getitem_raises_with_the_declared_names(aero_outputs):
    """Looking up an undeclared name reports what is declared, since it is usually a typo."""
    with pytest.raises(KeyError) as excinfo:
        aero_outputs["ac|aero|CLmax_to"]
    assert "ac|aero|CLmax_TO" in str(excinfo.value)


def test_set_deduplicates_identical_declarations():
    """Declaring the same variable twice identically collapses to one entry."""
    duplicated = VariableSet([Variable("weight", "kg", vectorized=True)] * 3)
    assert len(duplicated) == 1


def test_set_construction_raises_on_conflicting_declarations():
    """A units disagreement inside one set is an error, not a last-one-wins resolution."""
    with pytest.raises(VariableConflictError, match="Conflicting declarations of 'weight'"):
        VariableSet([Variable("weight", "kg"), Variable("weight", "lbm")])


def test_union_raises_on_conflict_rather_than_choosing():
    """Merging two disciplines that disagree on units fails at assembly time."""
    metric = VariableSet([Variable("thrust", "N", vectorized=True)])
    imperial = VariableSet([Variable("thrust", "lbf", vectorized=True)])
    with pytest.raises(VariableConflictError, match="'thrust'"):
        metric | imperial


def test_union_intersection_difference(aero_outputs):
    """Set algebra returns new sets with the expected contents."""
    prop = VariableSet(
        [
            Variable("thrust", "N", vectorized=True),
            Variable("ac|aero|CLmax_TO", None),
        ]
    )

    assert (aero_outputs | prop).names == {"drag", "thrust", "ac|aero|CLmax_TO"}
    assert (aero_outputs & prop).names == {"ac|aero|CLmax_TO"}
    assert (aero_outputs - prop).names == {"drag"}


def test_set_operations_do_not_mutate_operands(aero_outputs):
    """Set algebra is non-destructive; the operands are unchanged afterwards."""
    before = aero_outputs.names
    other = VariableSet([Variable("thrust", "N", vectorized=True)])
    aero_outputs | other
    aero_outputs - other
    aero_outputs & other
    assert aero_outputs.names == before
    assert other.names == {"thrust"}


def test_set_is_immutable(aero_outputs):
    """A set rejects attribute assignment, so it can be shared without defensive copying."""
    with pytest.raises(AttributeError, match="immutable"):
        aero_outputs._by_name = {}
    assert len(aero_outputs) == 2


def test_empty_set_is_falsy_by_length_and_iterates_empty():
    """An empty set is well defined and safe to union against."""
    empty = VariableSet()
    assert len(empty) == 0
    assert list(empty) == []
    assert (empty | VariableSet([Variable("drag", "N")])).names == {"drag"}


def test_set_equality_and_hash_ignore_insertion_order():
    """Two sets holding the same declarations are equal regardless of order."""
    a = VariableSet([Variable("drag", "N"), Variable("thrust", "N")])
    b = VariableSet([Variable("thrust", "N"), Variable("drag", "N")])
    assert a == b
    assert hash(a) == hash(b)
