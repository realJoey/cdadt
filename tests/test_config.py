"""Unit: the case file is validated hard, because a bad key in a file fails late or never."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from cdadt import Config, ConfigError, DesignVariableSpec, ObjectiveSpec, PhaseSchedule
from cdadt.config import MissionConfig, RequirementSpec
from cdadt.mission import STEADY_FLIGHT_PHASES

MINIMAL = {
    "black_box": {"model": "some.module:SomeClass", "num_nodes": 11},
    "aircraft": {"ac|geom|wing|S_ref": {"value": 124.6, "units": "m**2", "source": "specs"}},
    "mission": {
        "parameters": {"mission_range": {"value": 2800, "units": "nmi"}},
        "schedule": {
            phase: {"Ueas": {"value": 250, "units": "kn"}, "vs": {"value": 0, "units": "ft/min"}}
            for phase in STEADY_FLIGHT_PHASES
        },
    },
}


def _case(**overrides):
    """Return a copy of the minimal case with the given top-level sections replaced."""
    data = copy.deepcopy(MINIMAL)
    data.update(copy.deepcopy(overrides))
    return data


# =============================================================================================
# Structure
# =============================================================================================


@pytest.mark.unit
def test_a_minimal_case_parses():
    """Only the optimization section is optional."""
    config = Config.from_dict(_case())
    assert config.black_box.model == "some.module:SomeClass"
    assert config.black_box.num_nodes == 11
    assert not config.is_optimization
    assert config.aircraft[0].name == "ac|geom|wing|S_ref"


@pytest.mark.unit
@pytest.mark.parametrize("section", ["black_box", "aircraft", "mission"])
def test_a_missing_required_section_is_named(section):
    """The error says which section, not that a key lookup failed."""
    data = _case()
    del data[section]
    with pytest.raises(ConfigError, match=f"requires a '{section}' key"):
        Config.from_dict(data)


@pytest.mark.unit
def test_an_unknown_top_level_key_is_refused():
    """A silently ignored key answers a different question than the one that was asked."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(_case(mision={}))


@pytest.mark.unit
def test_an_unknown_key_inside_a_section_is_refused():
    """The same reasoning, one level down."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(_case(solver={"maxiter": 20, "maxiters": 30}))


@pytest.mark.unit
def test_a_parameter_without_a_value_is_refused():
    """Units without a value is a half-written entry."""
    with pytest.raises(ConfigError, match="requires a 'value' key"):
        Config.from_dict(_case(aircraft={"ac|geom|wing|AR": {"units": "m**2"}}))


@pytest.mark.unit
def test_a_non_numeric_parameter_value_is_refused():
    """Caught in the case file, where the offending key can be named."""
    with pytest.raises(ConfigError, match="non-numeric"):
        Config.from_dict(_case(aircraft={"ac|geom|wing|AR": {"value": "wide"}}))


# =============================================================================================
# Mission
# =============================================================================================


@pytest.mark.unit
def test_every_steady_flight_phase_must_be_scheduled():
    """An unscheduled phase flies its component's placeholder values."""
    data = _case()
    del data["mission"]["schedule"]["loiter"]
    with pytest.raises(ConfigError, match="missing a schedule for"):
        Config.from_dict(data)


@pytest.mark.unit
def test_a_phase_needs_both_speed_and_vertical_speed():
    """Setting one alone flies a profile that is half inherited from whatever came before."""
    data = _case()
    data["mission"]["schedule"]["cruise"] = {"Ueas": {"value": 250}}
    with pytest.raises(ConfigError, match="both 'Ueas' and 'vs' are required"):
        Config.from_dict(data)


@pytest.mark.unit
def test_a_phase_the_mission_does_not_fly_is_refused():
    """A typo in a phase name would otherwise schedule nothing at all."""
    data = _case()
    data["mission"]["schedule"]["crusie"] = {
        "Ueas": {"value": 250},
        "vs": {"value": 0},
    }
    with pytest.raises(ConfigError, match="does not fly"):
        Config.from_dict(data)


@pytest.mark.unit
def test_a_continuation_step_may_override_only_some_phases():
    """Steps are partial by design; the design mission is what must be complete."""
    data = _case()
    data["mission"]["continuation"] = [
        {
            "description": "easier",
            "parameters": {"mission_range": {"value": 500, "units": "nmi"}},
            "schedule": {"descent": {"Ueas": {"value": 250}, "vs": {"value": -800}}},
        }
    ]
    config = Config.from_dict(data)
    profile = config.mission.profile()
    assert len(profile.continuation) == 1
    assert profile.continuation[0].description == "easier"
    assert set(profile.continuation[0].schedules) == {"descent"}


@pytest.mark.unit
def test_mission_parameter_names_carry_the_mission_prefix():
    """They are checked against the black box, which publishes them under the mission group."""
    config = Config.from_dict(_case())
    assert config.mission.parameter_names == ("mission.mission_range",)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("given", "expected"),
    [(250.0, [250.0] * 5), ([230, 270], [230.0, 240.0, 250.0, 260.0, 270.0])],
)
def test_a_schedule_is_resampled_onto_the_grid(given, expected):
    """One constant, two endpoints to interpolate, or one value per node."""
    schedule = PhaseSchedule(given, 0.0)
    assert schedule.airspeed_at(5) == pytest.approx(np.asarray(expected))


@pytest.mark.unit
def test_a_schedule_of_the_wrong_length_is_refused():
    """Broadcasting it would quietly fly a different mission."""
    with pytest.raises(ValueError, match="has 3 values but the mission has 5 nodes"):
        PhaseSchedule([1.0, 2.0, 3.0], 0.0).airspeed_at(5)


@pytest.mark.unit
def test_a_mission_profile_needs_every_phase():
    """Enforced by the profile itself, not only by the case-file reader."""
    with pytest.raises(ValueError, match="No airspeed/vertical-speed schedule"):
        MissionConfig(parameters={}, schedules={"cruise": PhaseSchedule(250.0, 0.0)}).profile()


# =============================================================================================
# Optimization
# =============================================================================================


@pytest.mark.unit
def test_an_optimization_section_parses():
    """Design variables, objective and requirements are all data."""
    data = _case(
        optimization={
            "driver": {"name": "SLSQP", "maxiter": 25, "tol": 1e-7, "derivative_mode": "fwd"},
            "objective": {"name": "total_fuel", "units": "kg", "ref": 2e4},
            "design_variables": [{"name": "ac|geom|wing|S_ref", "lower": 90, "upper": 180, "units": "m**2"}],
            "requirements": [
                {
                    "type": "balanced_field_length",
                    "limit": 8000,
                    "units": "ft",
                    "regulation": "14 CFR 25.113",
                    "source": "8000 ft dry runway",
                }
            ],
        }
    )
    config = Config.from_dict(data)
    assert config.is_optimization
    settings = config.optimization
    assert settings.driver == "SLSQP"
    assert settings.maxiter == 25
    assert settings.design_variables[0].name == "ac|geom|wing|S_ref"
    assert settings.requirements[0].regulation == "14 CFR 25.113"


@pytest.mark.unit
def test_an_optimization_needs_a_design_variable():
    """Nothing to change is not an optimization."""
    with pytest.raises(ConfigError, match="at least one design variable"):
        Config.from_dict(
            _case(
                optimization={
                    "objective": {"name": "total_fuel"},
                    "design_variables": [],
                }
            )
        )


@pytest.mark.unit
def test_a_design_variable_declared_twice_is_refused():
    """The second declaration would silently replace the first."""
    with pytest.raises(ConfigError, match="declared more than once"):
        Config.from_dict(
            _case(
                optimization={
                    "objective": {"name": "total_fuel"},
                    "design_variables": [
                        {"name": "ac|geom|wing|AR", "lower": 7, "upper": 13},
                        {"name": "ac|geom|wing|AR", "lower": 8, "upper": 12},
                    ],
                }
            )
        )


@pytest.mark.unit
def test_inverted_design_variable_bounds_are_refused():
    """An empty interval is a typo, and the optimizer's own message would not say so."""
    with pytest.raises(ConfigError, match=r"lower .* >= upper"):
        DesignVariableSpec("ac|geom|wing|AR", lower=13.0, upper=7.0)


@pytest.mark.unit
def test_a_design_variable_scales_itself_from_its_bounds_by_default():
    """Wing area in square metres and aspect ratio must be comparable to the optimizer."""
    assert DesignVariableSpec("ac|geom|wing|S_ref", 90.0, 180.0).ref == 180.0
    assert DesignVariableSpec("ac|geom|wing|S_ref", 90.0, 180.0, ref=100.0).ref == 100.0


@pytest.mark.unit
def test_an_unknown_objective_sense_is_refused():
    """There are two, and a third would be silently treated as one of them."""
    with pytest.raises(ConfigError, match="sense must be one of"):
        ObjectiveSpec("total_fuel", sense="lower")


@pytest.mark.unit
def test_maximizing_negates_the_objective_reference():
    """OpenMDAO drivers always minimize; the sign lives in one place."""
    assert ObjectiveSpec("range", sense="minimize").scaler == 1.0
    assert ObjectiveSpec("range", sense="maximize").scaler == -1.0


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["regulation", "source"])
def test_a_requirement_must_name_its_regulation_and_its_source(missing):
    """A constraint nobody can defend has no place in a certification-driven study."""
    entry = {
        "type": "balanced_field_length",
        "limit": 8000,
        "regulation": "14 CFR 25.113",
        "source": "8000 ft dry runway",
    }
    del entry[missing]
    with pytest.raises(ConfigError, match=f"requires a '{missing}' key"):
        RequirementSpec.from_dict(entry, "optimization.requirements[0]")


@pytest.mark.unit
def test_an_unknown_derivative_mode_is_refused():
    """OpenMDAO would accept it and mean something else."""
    with pytest.raises(ConfigError, match="derivative_mode"):
        Config.from_dict(
            _case(
                optimization={
                    "driver": {"derivative_mode": "forward"},
                    "objective": {"name": "total_fuel"},
                    "design_variables": [{"name": "ac|geom|wing|AR", "lower": 7, "upper": 13}],
                }
            )
        )


# =============================================================================================
# The shipped cases
# =============================================================================================


@pytest.mark.unit
def test_the_shipped_sizing_case_parses(sizing_config):
    """The case a user is most likely to copy must be valid."""
    assert sizing_config.black_box.num_nodes == 21
    assert len(sizing_config.aircraft) == 34
    assert len(sizing_config.mission.profile().continuation) == 2
    assert not sizing_config.is_optimization


@pytest.mark.unit
def test_the_shipped_optimization_case_parses(optimization_config):
    """Including its certification basis."""
    settings = optimization_config.optimization
    assert settings is not None
    assert len(settings.design_variables) == 5
    assert len(settings.requirements) == 4
    assert {requirement.kind for requirement in settings.requirements} == {
        "balanced_field_length",
        "engine_out_climb_gradient",
        "throttle_limit",
    }


@pytest.mark.unit
def test_every_aircraft_parameter_in_the_shipped_cases_records_a_source(sizing_config, optimization_config):
    """A number with no provenance is indistinguishable from a guess in the report quoting it."""
    for config in (sizing_config, optimization_config):
        unsourced = [parameter.name for parameter in config.aircraft if not parameter.source]
        assert not unsourced, f"{config.path} has unsourced parameters: {unsourced}"


@pytest.mark.unit
def test_the_two_shipped_cases_describe_the_same_aeroplane(sizing_config, optimization_config):
    """The optimization case adds a question; it must not quietly change the aircraft."""
    sizing = {parameter.name: (parameter.value, parameter.units) for parameter in sizing_config.aircraft}
    optimizing = {parameter.name: (parameter.value, parameter.units) for parameter in optimization_config.aircraft}
    assert sizing == optimizing


@pytest.mark.unit
def test_the_shipped_cases_are_valid_yaml_documents():
    """Parsed independently of cdadt, so a YAML error is reported as one."""
    from tests.conftest import CASES

    for path in sorted(CASES.glob("*.yaml")):
        with open(path, encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), dict), path


# =============================================================================================
# Error branches and representations
# =============================================================================================


@pytest.mark.unit
def test_a_section_that_is_not_a_mapping_says_so():
    """A YAML list where a mapping was expected fails by name rather than by AttributeError."""
    with pytest.raises(ConfigError, match="must be a mapping"):
        Config.from_dict(_case(black_box=["model", "num_nodes"]))


@pytest.mark.unit
def test_design_variables_and_requirements_must_be_lists():
    """A mapping there is a plausible mistake that would otherwise iterate over its keys."""
    base = {"objective": {"name": "total_fuel"}}
    with pytest.raises(ConfigError, match="design_variables' must be a list"):
        Config.from_dict(_case(optimization={**base, "design_variables": {"name": "ac|geom|wing|AR"}}))
    with pytest.raises(ConfigError, match="requirements' must be a list"):
        Config.from_dict(
            _case(
                optimization={
                    **base,
                    "design_variables": [{"name": "ac|geom|wing|AR", "lower": 7, "upper": 13}],
                    "requirements": {"type": "balanced_field_length"},
                }
            )
        )


@pytest.mark.unit
def test_a_non_numeric_mission_parameter_is_refused():
    """Mission values go through the same quantity reader as aircraft parameters."""
    data = _case()
    data["mission"]["parameters"] = {"mission_range": {"value": "far", "units": "nmi"}}
    with pytest.raises(ConfigError, match="must be a number"):
        Config.from_dict(data)


@pytest.mark.unit
def test_a_file_that_is_not_valid_yaml_is_reported_as_such(tmp_path):
    """Distinguished from a valid file with wrong contents, which is a different fix."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("black_box: {model: x\n  num_nodes: 11\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid YAML"):
        Config.from_yaml(broken)


@pytest.mark.unit
def test_a_file_whose_top_level_is_not_a_mapping_is_refused(tmp_path):
    """A list of cases is a plausible thing to write, and is not what this reader takes."""
    listy = tmp_path / "listy.yaml"
    listy.write_text("- black_box\n- mission\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a mapping at the top level"):
        Config.from_yaml(listy)


@pytest.mark.unit
def test_the_mission_section_exposes_its_path():
    """The mission subsystem name is configurable, and is what parameter names are prefixed with."""
    data = _case()
    data["mission"]["mission_path"] = "trajectory"
    config = Config.from_dict(data)
    assert config.mission.mission_path == "trajectory"
    assert config.mission.parameter_names == ("trajectory.mission_range",)


@pytest.mark.unit
def test_every_configuration_object_reprs_as_what_it_holds(tmp_path):
    """Reprs are what a debugger frame shows while a case is being diagnosed."""
    data = _case(
        solver={"maxiter": 25},
        optimization={
            "objective": {"name": "total_fuel", "sense": "minimize"},
            "design_variables": [{"name": "ac|geom|wing|AR", "lower": 7, "upper": 13}],
            "requirements": [
                {
                    "type": "balanced_field_length",
                    "limit": 8000,
                    "regulation": "14 CFR 25.113",
                    "source": "an 8000 ft runway",
                }
            ],
        },
    )
    config = Config.from_dict(data)
    assert repr(config.black_box) == "BlackBoxConfig('some.module:SomeClass', num_nodes=11)"
    assert repr(config.solver) == "SolverConfig(maxiter=25)"
    assert repr(config.mission).startswith("MissionConfig(7 phases")
    assert repr(config.optimization).startswith("OptimizationConfig(objective='total_fuel'")
    assert repr(config.optimization.objective) == "ObjectiveSpec('total_fuel', sense='minimize')"
    assert repr(config.optimization.design_variables[0]).startswith("DesignVariableSpec('ac|geom|wing|AR'")
    assert repr(config.optimization.requirements[0]).startswith("RequirementSpec('balanced_field_length'")
    assert repr(config) == "Config(in memory, optimization, 1 parameters)"

    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert repr(Config.from_yaml(path)) == f"Config({path}, sizing, 1 parameters)"
