"""Unit: the two value objects every discipline is built from."""

from __future__ import annotations

import pytest

from cdadt import Parameter, Response


@pytest.mark.unit
def test_a_parameter_keeps_its_value_units_and_source():
    """The three things a design number needs to be arguable."""
    area = Parameter("ac|geom|wing|S_ref", 124.6, "m**2", source="b737.org.uk")
    assert area.name == "ac|geom|wing|S_ref"
    assert area.value == 124.6
    assert area.units == "m**2"
    assert area.source == "b737.org.uk"


@pytest.mark.unit
def test_a_parameter_value_can_be_changed_because_an_optimizer_changes_it():
    """The only mutable state in the class, and it is mutable for one reason."""
    aspect_ratio = Parameter("ac|geom|wing|AR", 9.45)
    aspect_ratio.value = 11.0
    assert aspect_ratio.value == 11.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_value_is_refused(bad):
    """A NaN reaching the box surfaces as a convergence failure with no explanation."""
    with pytest.raises(ValueError, match="non-finite"):
        Parameter("ac|geom|wing|AR", bad)


@pytest.mark.unit
def test_a_non_numeric_value_is_refused():
    """A string that looks like a number in YAML is still a mistake worth naming."""
    with pytest.raises(ValueError, match="non-numeric"):
        Parameter("ac|geom|wing|AR", "wide")


@pytest.mark.unit
def test_a_parameter_needs_a_name():
    """An unnamed parameter cannot be routed to a discipline or set on the box."""
    with pytest.raises(ValueError, match="must have a name"):
        Parameter("", 1.0)


@pytest.mark.unit
def test_replace_copies_everything_but_the_value():
    """Used when a value changes but its identity and provenance do not."""
    original = Parameter("ac|geom|wing|AR", 9.45, source="specs")
    copy = original.replace(11.0)
    assert (copy.name, copy.units, copy.source) == (original.name, original.units, original.source)
    assert copy.value == 11.0
    assert original.value == 9.45


@pytest.mark.unit
def test_a_response_needs_a_name_and_a_path():
    """A response with no path is a quantity nothing can read."""
    with pytest.raises(ValueError, match="must have a name"):
        Response("", "some.path")
    with pytest.raises(ValueError, match="must name a path"):
        Response("total_fuel", "")


@pytest.mark.unit
def test_responses_compare_on_what_they_mean():
    """Two responses are the same response if they read the same thing the same way."""
    one = Response("total_fuel", "mission.loiter.fuel_burn_integ.fuel_burn_final", "kg", "block plus reserves")
    two = Response("total_fuel", "mission.loiter.fuel_burn_integ.fuel_burn_final", "kg", "different words")
    three = Response("total_fuel", "mission.descent.fuel_burn_integ.fuel_burn_final", "kg")
    assert one == two
    assert one != three
