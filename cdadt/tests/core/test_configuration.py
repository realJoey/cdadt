"""Tests for :mod:`cdadt.core.configuration`.

Claim class: ``unit``. These prove the configuration object validates its contents, converts
units through OpenMDAO's real unit system, and -- most importantly -- has no mechanism for
producing a value that was not configured.

That last property is the reason this class exists. A method constant supplied as a default
is a hardcoded number that happens to carry a citation, and a model that can fall back to
one will silently produce a result nobody chose. The tests below assert that every read path
raises on an absent entry, including the paths a future contributor might reach for.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdadt.core.configuration import (
    AircraftConfiguration,
    ConfigurationError,
    MissingConfigurationError,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def config():
    """Return a small configuration in the OpenConcept nested-dictionary format."""
    return AircraftConfiguration(
        {
            "ac": {
                "geom": {
                    "wing": {
                        "S_ref": {"value": 124.6, "units": "m**2"},
                        "AR": {"value": 9.45},
                        "toverc": {"value": [0.15, 0.12, 0.10]},
                    },
                    "fuselage": {"length": {"value": 38.08, "units": "m"}},
                },
                "weights": {"W_payload": {"value": 18e3, "units": "kg"}},
            },
            "certification": {
                "landing": {
                    "approach_speed_factor": {
                        "value": 1.23,
                        "source": "14 CFR 25.125(a)(2), V_REF >= 1.23 V_SR0",
                    }
                }
            },
        }
    )


# ==============================================================================
# Construction and validation
# ==============================================================================
def test_names_are_flattened_pipe_separated_paths(config):
    """Every leaf is addressable by the OpenConcept-style pipe-separated path."""
    assert config.names == (
        "ac|geom|fuselage|length",
        "ac|geom|wing|AR",
        "ac|geom|wing|S_ref",
        "ac|geom|wing|toverc",
        "ac|weights|W_payload",
        "certification|landing|approach_speed_factor",
    )


def test_rejects_non_mapping_input():
    """A configuration must be built from a mapping."""
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        AircraftConfiguration([("ac", 1)])


def test_rejects_leaf_without_a_value_key():
    """A branch that bottoms out without a 'value' key is malformed, not an empty branch."""
    with pytest.raises(ConfigurationError, match="must be a mapping or a"):
        AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": 124.6}}}})


def test_rejects_leaf_with_invalid_units():
    """Unit strings are validated against OpenMDAO at load time, not at solve time."""
    with pytest.raises(ConfigurationError, match="declares invalid units 'furlongs'"):
        AircraftConfiguration({"ac": {"geom": {"span": {"value": 35.0, "units": "furlongs"}}}})


def test_rejects_leaf_with_non_numeric_value():
    """A configuration holds numbers; a string value is a mistake worth catching early."""
    with pytest.raises(ConfigurationError, match="non-numeric value of type str"):
        AircraftConfiguration({"ac": {"geom": {"span": {"value": "thirty five", "units": "m"}}}})


def test_rejects_leaf_with_unexpected_keys():
    """Only value, units and source are permitted, so a typo'd key cannot be ignored."""
    with pytest.raises(ConfigurationError, match=r"unexpected keys \['unit'\]"):
        AircraftConfiguration({"ac": {"geom": {"span": {"value": 35.0, "unit": "m"}}}})


def test_rejects_empty_branch():
    """An empty branch means an entry was intended and never filled in."""
    with pytest.raises(ConfigurationError, match="empty branch"):
        AircraftConfiguration({"ac": {"geom": {}}})


# ==============================================================================
# The no-defaults rule
# ==============================================================================
def test_value_raises_on_absent_entry(config):
    """Reading an unconfigured name raises rather than returning anything."""
    with pytest.raises(MissingConfigurationError) as excinfo:
        config.value("ac|geom|wing|c4sweep", units="deg")
    assert "does not supply fallback values" in str(excinfo.value)


def test_value_has_no_default_parameter(config):
    """There is no ``default`` keyword to reach for.

    This is asserted on the signature rather than on behavior: the rule is that the
    capability does not exist, so adding it would have to break this test deliberately.
    """
    import inspect

    parameters = inspect.signature(config.value).parameters
    assert "default" not in parameters
    assert "fallback" not in parameters
    assert set(parameters) == {"name", "units"}


def test_missing_entry_error_suggests_nearby_names(config):
    """The error names configured entries sharing the requested path prefix."""
    with pytest.raises(MissingConfigurationError) as excinfo:
        config.value("ac|geom|wing|c4sweep")
    message = str(excinfo.value)
    assert "ac|geom|wing|AR" in message or "ac|geom|wing|S_ref" in message


def test_require_all_reports_every_missing_entry_at_once(config):
    """A provider validating its constants sees the full list, not one per traceback."""
    with pytest.raises(MissingConfigurationError) as excinfo:
        config.require_all(
            [
                "ac|geom|wing|S_ref",
                "ac|geom|wing|c4sweep",
                "ac|aero|polar|e",
                "ac|propulsion|num_engines",
            ]
        )
    message = str(excinfo.value)
    assert "ac|geom|wing|c4sweep" in message
    assert "ac|aero|polar|e" in message
    assert "ac|propulsion|num_engines" in message
    assert "3 required configuration entries" in message


def test_require_all_passes_when_everything_is_present(config):
    """require_all returns quietly when the configuration is complete."""
    config.require_all(["ac|geom|wing|S_ref", "ac|weights|W_payload"])


def test_intermediate_branch_is_not_readable_as_a_value(config):
    """Addressing a branch rather than a leaf raises, instead of returning a dict."""
    with pytest.raises(MissingConfigurationError):
        config.value("ac|geom|wing")


# ==============================================================================
# Reading values
# ==============================================================================
def test_value_returns_an_array_even_for_scalars(config):
    """Values come back as arrays so callers handle scalars and vectors identically."""
    value = config.value("ac|geom|wing|S_ref")
    assert isinstance(value, np.ndarray)
    assert value.shape == (1,)
    assert value[0] == pytest.approx(124.6)


def test_value_preserves_vector_entries(config):
    """A list-valued entry keeps its length."""
    assert config.value("ac|geom|wing|toverc") == pytest.approx([0.15, 0.12, 0.10])


def test_value_converts_units_through_openmdao(config):
    """Unit conversion uses OpenMDAO's unit system, matching what the model will do.

    124.6 m^2 is 1341.185... ft^2; the expected value is computed from the exact
    conversion factor rather than transcribed, so the test does not encode a rounded
    constant of its own.
    """
    expected = 124.6 / (0.3048**2)
    assert config.value("ac|geom|wing|S_ref", units="ft**2")[0] == pytest.approx(expected, rel=1e-12)


def test_value_returns_configured_units_when_none_requested(config):
    """Omitting units returns the number as configured, unconverted."""
    assert config.value("ac|geom|fuselage|length")[0] == pytest.approx(38.08)


def test_requesting_units_for_a_dimensionless_entry_raises(config):
    """Asking for units on a dimensionless entry is a modeling error, not a no-op."""
    with pytest.raises(ConfigurationError, match="configured as dimensionless"):
        config.value("ac|geom|wing|AR", units="m")


def test_incompatible_unit_conversion_raises(config):
    """Converting an area to a length reports the offending entry by name."""
    with pytest.raises(ConfigurationError, match="ac\\|geom\\|wing\\|S_ref"):
        config.value("ac|geom|wing|S_ref", units="kg")


def test_scalar_returns_a_float(config):
    """scalar() is the convenience path for single-valued entries."""
    assert config.scalar("ac|weights|W_payload", units="kg") == pytest.approx(18e3)


def test_scalar_raises_on_a_vector_entry(config):
    """Silently taking the first element of a vector would hide a configuration error."""
    with pytest.raises(ConfigurationError, match="holds 3 elements"):
        config.scalar("ac|geom|wing|toverc")


def test_units_and_source_are_readable(config):
    """Units and provenance are retrievable for the traceability report."""
    assert config.units("ac|geom|wing|S_ref") == "m**2"
    assert config.units("ac|geom|wing|AR") is None
    assert config.source("certification|landing|approach_speed_factor") == ("14 CFR 25.125(a)(2), V_REF >= 1.23 V_SR0")
    assert config.source("ac|geom|wing|S_ref") is None


# ==============================================================================
# Immutability and derived configurations
# ==============================================================================
def test_configuration_rejects_attribute_assignment(config):
    """A configuration is a fixed record of one design."""
    with pytest.raises(AttributeError, match="immutable"):
        config._data = {}


def test_as_dict_returns_a_copy_that_cannot_reach_back(config):
    """Handing the dictionary to OpenConcept cannot let the consumer mutate the source."""
    exported = config.as_dict()
    exported["ac"]["geom"]["wing"]["S_ref"]["value"] = 1.0
    assert config.scalar("ac|geom|wing|S_ref") == pytest.approx(124.6)


def test_with_overrides_returns_a_new_configuration_and_inherits_units(config):
    """Overriding a raw value keeps the configured units and leaves the original alone."""
    modified = config.with_overrides({"ac|geom|wing|S_ref": 150.0})
    assert modified.scalar("ac|geom|wing|S_ref", units="m**2") == pytest.approx(150.0)
    assert config.scalar("ac|geom|wing|S_ref", units="m**2") == pytest.approx(124.6)


def test_with_overrides_accepts_a_full_leaf_for_a_new_entry(config):
    """A new entry may be added, but only with explicit units."""
    modified = config.with_overrides({"ac|geom|wing|c4sweep": {"value": 25.0, "units": "deg"}})
    assert modified.scalar("ac|geom|wing|c4sweep", units="deg") == pytest.approx(25.0)


def test_with_overrides_rejects_a_bare_value_for_an_unconfigured_name(config):
    """A raw value for a new name has no units to inherit, so it is refused."""
    with pytest.raises(MissingConfigurationError):
        config.with_overrides({"ac|geom|wing|c4sweep": 25.0})


def test_subtree_keeps_full_names(config):
    """A discipline's slice of the configuration is addressed by the same full names."""
    geom = config.subtree("ac|geom")
    assert geom.names == ("ac|geom|fuselage|length", "ac|geom|wing|AR", "ac|geom|wing|S_ref", "ac|geom|wing|toverc")
    assert geom.scalar("ac|geom|wing|S_ref", units="m**2") == pytest.approx(124.6)
    assert "ac|weights|W_payload" not in geom


def test_subtree_raises_when_the_prefix_matches_nothing(config):
    """An empty subtree means the discipline's configuration section is absent."""
    with pytest.raises(MissingConfigurationError):
        config.subtree("ac|propulsion")


# ==============================================================================
# YAML loading
# ==============================================================================
def test_from_yaml_round_trips(tmp_path):
    """A YAML file loads into the same structure as the equivalent dictionary."""
    path = tmp_path / "aircraft.yml"
    path.write_text(
        "ac:\n"
        "  geom:\n"
        "    wing:\n"
        "      S_ref:\n"
        "        value: 124.6\n"
        "        units: m**2\n"
        "      AR:\n"
        "        value: 9.45\n",
        encoding="utf-8",
    )
    loaded = AircraftConfiguration.from_yaml(path)
    assert loaded.scalar("ac|geom|wing|S_ref", units="m**2") == pytest.approx(124.6)
    assert loaded.scalar("ac|geom|wing|AR") == pytest.approx(9.45)


def test_from_yaml_rejects_a_non_mapping_document(tmp_path):
    """A YAML list at the top level is not a configuration."""
    path = tmp_path / "bad.yml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping at the top level"):
        AircraftConfiguration.from_yaml(path)


def test_from_yaml_propagates_leaf_validation(tmp_path):
    """Validation applies to YAML input exactly as it does to dictionary input."""
    path = tmp_path / "bad_units.yml"
    path.write_text("ac:\n  geom:\n    span:\n      value: 35.0\n      units: furlongs\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid units"):
        AircraftConfiguration.from_yaml(path)
