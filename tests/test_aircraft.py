"""Unit tests for :class:`~cdadt.aircraft.AircraftDefinition`."""

from __future__ import annotations

import openmdao.api as om
import pytest

from cdadt import AircraftDefinition

pytestmark = pytest.mark.unit


def test_leaves_are_found_and_named_with_pipes(aircraft):
    """A leaf is a mapping with a 'value' key, named by its path."""
    assert "ac|geom|wing|S_ref" in aircraft
    assert "ac|geom|wing" not in aircraft  # a group is not a parameter
    assert aircraft.value("ac|geom|wing|S_ref") == pytest.approx(124.6)
    assert aircraft.units("ac|geom|wing|S_ref") == "m**2"
    assert aircraft.units("ac|geom|wing|AR") is None


def test_names_are_sorted_and_complete(aircraft):
    """Every leaf appears exactly once, in a stable order."""
    assert aircraft.names == tuple(sorted(aircraft.names))
    assert len(aircraft.names) == len(set(aircraft.names))
    assert len(aircraft) == len(aircraft.names)


def test_the_definition_is_copied_not_aliased():
    """Two definitions built from one literal do not share state."""
    data = {"ac": {"geom": {"wing": {"AR": {"value": 9.0}}}}}
    first, second = AircraftDefinition(data), AircraftDefinition(data)
    first.set("ac|geom|wing|AR", 12.0)
    assert second.value("ac|geom|wing|AR") == pytest.approx(9.0)
    assert data["ac"]["geom"]["wing"]["AR"]["value"] == pytest.approx(9.0)


def test_setting_an_undefined_parameter_raises(aircraft):
    """A typo must not silently create a variable nothing reads."""
    with pytest.raises(KeyError, match="not defined"):
        aircraft.set("ac|geom|wing|AR_typo", 10.0)


def test_reading_a_group_raises(aircraft):
    """Naming a subtree is an error, not an empty result."""
    with pytest.raises(KeyError, match="group of parameters"):
        aircraft.value("ac|geom|wing")


def test_an_empty_definition_raises():
    """A definition with no leaves is a configuration mistake, not an empty aircraft."""
    with pytest.raises(ValueError, match="no parameters"):
        AircraftDefinition({"ac": {"geom": {}}})


def test_a_non_numeric_value_raises():
    """YAML that fails to parse a number must not reach OpenMDAO as a string."""
    with pytest.raises(ValueError, match="non-numeric"):
        AircraftDefinition({"ac": {"geom": {"wing": {"AR": {"value": "9.45e3"}}}}})


def test_the_component_publishes_every_parameter(aircraft):
    """Each parameter becomes an independent variable with its declared units."""
    problem = om.Problem(reports=False)
    problem.model.add_subsystem("parameters", aircraft.component(), promotes_outputs=["*"])
    problem.setup(check=False)
    problem.run_model()

    for name in aircraft.names:
        assert problem.get_val(name, units=aircraft.units(name)).item() == pytest.approx(aircraft.value(name))
