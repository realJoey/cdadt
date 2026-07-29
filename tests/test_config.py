"""Unit: the case file is validated hard, because a bad key in a file fails late or never."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from cdadt import Bounds, CaseFileSection, Config, ConfigError, ObjectiveSpec, OptimizeSpec, Scaling, VariableSpec
from cdadt.config import ABSENT

MINIMAL = {
    "black_box": {"model": "some.module:SomeClass", "num_nodes": 11},
    "design_variables": {
        "ac|geom|wing|S_ref": {"value": 124.6, "units": "m**2", "source": "specs"},
    },
    "initial_conditions": {
        "mission_range": {"value": 2800, "units": "nmi"},
        "climb.fltcond|Ueas": {"value": [230, 252], "units": "kn"},
    },
}

OPTIMIZATION = {
    "driver": {"name": "SLSQP", "maxiter": 25, "tol": 1e-7, "derivative_mode": "fwd"},
    "objective": {"name": "total_fuel", "units": "kg", "ref": 2e4},
    "constraints": [
        {
            "name": "takeoff_field_length",
            "upper": 8000,
            "units": "ft",
            "regulation": "14 CFR 25.113",
            "source": "8000 ft dry runway",
        }
    ],
}


def _case(**overrides):
    """Return a copy of the minimal case with the given top-level sections replaced or added."""
    data = copy.deepcopy(MINIMAL)
    data.update(copy.deepcopy(overrides))
    return data


def _optimizing(**overrides):
    """Return a minimal case that also declares an objective and a free design variable."""
    data = _case(**copy.deepcopy(OPTIMIZATION))
    data["design_variables"]["ac|geom|wing|S_ref"]["optimize"] = {"lower": 90.0, "upper": 180.0}
    data.update(copy.deepcopy(overrides))
    return data


# =============================================================================================
# Structure
# =============================================================================================


@pytest.mark.unit
def test_a_minimal_case_parses():
    """Only ``black_box`` and ``design_variables`` are required."""
    config = Config.from_dict(_case())
    assert config.black_box.model == "some.module:SomeClass"
    assert config.black_box.num_nodes == 11
    assert not config.is_optimization
    assert "ac|geom|wing|S_ref" in config.design_variables


@pytest.mark.unit
@pytest.mark.parametrize("section", ["black_box", "design_variables"])
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
        Config.from_dict(_case(desing_variables={}))


@pytest.mark.unit
def test_an_unknown_key_inside_a_section_is_refused():
    """The same reasoning, one level down."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(_case(solver={"maxiter": 20, "maxiters": 30}))


@pytest.mark.unit
def test_an_unknown_key_on_a_variable_is_refused():
    """``optimise`` is a plausible misspelling that would silently leave a variable fixed."""
    data = _case()
    data["design_variables"]["ac|geom|wing|S_ref"]["optimise"] = {"lower": 1, "upper": 2}
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(data)


@pytest.mark.unit
def test_a_variable_without_a_value_is_refused():
    """Units without a value is a half-written entry."""
    with pytest.raises(ConfigError, match="requires a 'value' key"):
        Config.from_dict(_case(design_variables={"ac|geom|wing|AR": {"units": "m**2"}}))


@pytest.mark.unit
def test_a_non_numeric_value_is_refused():
    """Caught in the case file, where the offending key can be named."""
    with pytest.raises(ConfigError, match="must be a number"):
        Config.from_dict(_case(design_variables={"ac|geom|wing|AR": {"value": "wide"}}))
    with pytest.raises(ConfigError, match="non-numeric entry"):
        Config.from_dict(_case(design_variables={"ac|geom|wing|AR": {"value": [1.0, "wide"]}}))


@pytest.mark.unit
def test_a_variable_declared_in_both_blocks_is_refused():
    """A variable is set once, in one place, or the second write silently wins."""
    data = _case()
    data["initial_conditions"]["ac|geom|wing|S_ref"] = {"value": 130.0, "units": "m**2"}
    with pytest.raises(ConfigError, match="appear in both"):
        Config.from_dict(data)


# =============================================================================================
# Design variables and initial conditions
# =============================================================================================


@pytest.mark.unit
def test_a_variable_is_fixed_until_it_carries_an_optimize_entry():
    """The one thing that makes a design variable free, and it is local to the variable."""
    config = Config.from_dict(_case())
    assert not config.design_variables["ac|geom|wing|S_ref"].is_free
    assert config.free_variables == {}

    config = Config.from_dict(_optimizing())
    spec = config.design_variables["ac|geom|wing|S_ref"]
    assert spec.is_free
    assert spec.optimize.lower == 90.0 and spec.optimize.upper == 180.0
    assert list(config.free_variables) == ["ac|geom|wing|S_ref"]


@pytest.mark.unit
def test_an_initial_condition_may_also_be_freed():
    """OpenConcept's B738_VLM_drag example optimizes ``cruise|h0``; so may a cdadt case."""
    data = _optimizing()
    data["initial_conditions"]["cruise|h0"] = {
        "value": 35000,
        "units": "ft",
        "optimize": {"lower": 25000, "upper": 41000},
    }
    config = Config.from_dict(data)
    assert set(config.free_variables) == {"ac|geom|wing|S_ref", "cruise|h0"}


@pytest.mark.unit
def test_a_design_variable_cannot_be_bounded_by_an_equality():
    """``equals`` bounds a constraint; a design variable moves between two sides."""
    data = _case()
    data["design_variables"]["ac|geom|wing|S_ref"]["optimize"] = {"equals": 100.0}
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(data)


@pytest.mark.unit
def test_an_empty_optimize_band_is_refused():
    """An empty interval is a typo the optimizer's own message would not explain."""
    with pytest.raises(ConfigError, match="band is empty"):
        OptimizeSpec(Bounds(lower=13.0, upper=7.0))


@pytest.mark.unit
def test_a_free_variable_scales_itself_from_its_bounds_by_default():
    """Wing area in square metres and aspect ratio must be comparable to the optimizer."""
    assert OptimizeSpec(Bounds(lower=90.0, upper=180.0)).ref == 180.0
    assert OptimizeSpec(Bounds(lower=90.0, upper=180.0), Scaling(ref=100.0)).ref == 100.0


@pytest.mark.unit
def test_a_variable_may_be_a_per_node_schedule():
    """Two values are endpoints; that is how ``np.linspace`` is written as data."""
    config = Config.from_dict(_case())
    schedule = config.initial_conditions_specs["climb.fltcond|Ueas"]
    assert np.asarray(schedule.value).tolist() == [230.0, 252.0]
    assert schedule.units == "kn"


@pytest.mark.unit
def test_both_blocks_are_written_into_the_box():
    """A value is a value whichever block declared it; the split is about what a reader sees."""
    conditions = Config.from_dict(_case()).initial_conditions()
    assert "ac|geom|wing|S_ref" in conditions
    assert "mission_range" in conditions
    assert len(conditions) == 3


# =============================================================================================
# Continuation
# =============================================================================================


@pytest.mark.unit
def test_a_continuation_rung_states_only_what_it_relaxes():
    """Everything it does not name stays at the design condition."""
    data = _case(
        continuation=[
            {
                "description": "easier",
                "initial_conditions": {"mission_range": {"value": 500, "units": "nmi"}},
            }
        ]
    )
    ladder = Config.from_dict(data).continuation
    assert len(ladder) == 1
    assert ladder.steps[0].description == "easier"
    assert set(ladder.steps[0].conditions) == {"mission_range"}


@pytest.mark.unit
def test_a_continuation_rung_rejects_unknown_keys():
    """``schedule`` was the old spelling; silently ignoring it would fly the design mission."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict(_case(continuation=[{"description": "x", "schedule": {}}]))


@pytest.mark.unit
def test_continuation_must_be_a_list():
    """A mapping there is a plausible mistake that would otherwise iterate over its keys."""
    with pytest.raises(ConfigError, match="must be a list"):
        Config.from_dict(_case(continuation={"description": "x"}))


# =============================================================================================
# Constraints and the objective
# =============================================================================================


@pytest.mark.unit
def test_a_constraint_may_be_one_sided_two_sided_or_an_equality():
    """Every form ``add_constraint`` accepts, because the examples use every one."""
    data = _optimizing(
        constraints=[
            {"name": "takeoff_field_length", "upper": 8000, "units": "ft"},
            {"name": "climb_throttle", "lower": 0.01, "upper": 1.05},
            {"name": "mission_range_flown", "equals": 2800, "units": "nmi"},
        ]
    )
    one, two, equality = Config.from_dict(data).constraints
    assert one.bounds.upper == 8000 and one.bounds.lower is None
    assert two.bounds.is_two_sided and two.bounds.describe() == "0.01 to 1.05"
    assert equality.bounds.is_equality and equality.bounds.describe() == "= 2800"


@pytest.mark.unit
def test_a_constraint_with_no_bound_at_all_is_refused():
    """A named quantity with no limit constrains nothing."""
    with pytest.raises(ConfigError, match="needs a 'lower', an 'upper' or an 'equals'"):
        Config.from_dict(_optimizing(constraints=[{"name": "takeoff_field_length", "units": "ft"}]))


@pytest.mark.unit
def test_an_equality_cannot_be_combined_with_an_inequality():
    """OpenMDAO would take one and drop the other."""
    with pytest.raises(ConfigError, match="either an equality or an inequality"):
        Config.from_dict(_optimizing(constraints=[{"name": "MTOW", "equals": 1.0, "upper": 2.0}]))


@pytest.mark.unit
def test_provenance_is_optional_and_reported_as_such():
    """A constraint that names a regulation and a source is traceable; one that does not is not."""
    data = _optimizing(
        constraints=[
            {
                "name": "takeoff_field_length",
                "upper": 8000,
                "regulation": "14 CFR 25.113",
                "source": "8000 ft dry runway",
            },
            {"name": "climb_throttle", "lower": 0.01, "upper": 1.05},
        ]
    )
    traceable, plain = Config.from_dict(data).constraints
    assert traceable.is_traceable
    assert not plain.is_traceable
    assert plain.regulation == "" and plain.source == ""
    # A title defaults to the name, so a report always has something to print.
    assert plain.title == "climb_throttle"


@pytest.mark.unit
def test_an_objective_with_nothing_to_move_is_refused():
    """An objective and no free variable is a study that cannot do anything."""
    data = _case(**copy.deepcopy(OPTIMIZATION))
    with pytest.raises(ConfigError, match="nothing for the driver to move"):
        Config.from_dict(data)


@pytest.mark.unit
def test_an_unknown_objective_sense_is_refused():
    """There are two, and a third would be silently treated as one of them."""
    with pytest.raises(ConfigError, match="sense must be one of"):
        ObjectiveSpec("total_fuel", sense="lower")


@pytest.mark.unit
def test_maximizing_negates_the_objective_scaling():
    """OpenMDAO drivers only minimize; the sign lives in one place."""
    assert ObjectiveSpec("range", sense="minimize").as_kwargs(None)["ref"] == 1.0
    assert ObjectiveSpec("range", sense="maximize").as_kwargs(None)["ref"] == -1.0
    maximized = ObjectiveSpec("range", scaling=Scaling(scaler=2.0), sense="maximize").as_kwargs(None)
    assert maximized["scaler"] == -2.0


@pytest.mark.unit
def test_ref_and_scaler_cannot_both_be_given():
    """They are two spellings of the same affine map, and OpenMDAO refuses the pair."""
    with pytest.raises(ConfigError, match="not both"):
        Scaling(ref=1.0, scaler=2.0)


@pytest.mark.unit
def test_an_unknown_derivative_mode_is_refused():
    """OpenMDAO would accept it and mean something else."""
    with pytest.raises(ConfigError, match="derivative_mode"):
        Config.from_dict(_optimizing(driver={"derivative_mode": "forward"}))


# =============================================================================================
# Files
# =============================================================================================


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
    listy.write_text("- black_box\n- design_variables\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a mapping at the top level"):
        Config.from_yaml(listy)


@pytest.mark.unit
def test_a_section_that_is_not_a_mapping_says_so():
    """A YAML list where a mapping was expected fails by name rather than by AttributeError."""
    with pytest.raises(ConfigError, match="must be a mapping"):
        Config.from_dict(_case(black_box=["model", "num_nodes"]))


@pytest.mark.unit
def test_every_configuration_object_reprs_as_what_it_holds(tmp_path):
    """Reprs are what a debugger frame shows while a case is being diagnosed."""
    config = Config.from_dict(_optimizing())
    assert repr(config.black_box) == "BlackBoxConfig('some.module:SomeClass', num_nodes=11)"
    assert repr(config.solver) == "SolverConfig(maxiter=20)"
    assert repr(config.driver) == "DriverConfig('SLSQP', maxiter=25)"
    assert repr(config.objective) == "ObjectiveSpec('total_fuel', sense='minimize')"
    assert repr(config.constraints[0]).startswith("ConstraintSpec('takeoff_field_length'")
    assert repr(config.design_variables["ac|geom|wing|S_ref"]).endswith("free)")
    assert repr(config.free_variables["ac|geom|wing|S_ref"].optimize) == "OptimizeSpec(90 to 180)"
    assert repr(Scaling(ref=2.0)) == "Scaling({'ref': 2.0})"
    assert repr(Bounds(upper=1.0)) == "Bounds(<= 1)"
    assert repr(config) == "Config(in memory, optimization, 1 design variables)"

    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert repr(Config.from_yaml(path)) == f"Config({path}, sizing, 1 design variables)"


@pytest.mark.unit
def test_a_variable_spec_becomes_a_routable_parameter():
    """Scalars route to a discipline; a per-node schedule goes straight into the box."""
    spec = VariableSpec("ac|geom|wing|S_ref", 124.6, "m**2", "specs")
    parameter = spec.parameter()
    assert (parameter.name, parameter.value, parameter.units, parameter.source) == (
        "ac|geom|wing|S_ref",
        124.6,
        "m**2",
        "specs",
    )
    assert [p.name for p in Config.from_dict(_case()).parameters()] == ["ac|geom|wing|S_ref"]


# =============================================================================================
# The shipped cases
# =============================================================================================


@pytest.mark.unit
def test_the_shipped_sizing_case_parses(sizing_config):
    """The case a user is most likely to copy must be valid."""
    assert sizing_config.black_box.num_nodes == 21
    assert len(sizing_config.design_variables) == 34
    assert not sizing_config.free_variables
    assert len(sizing_config.continuation) == 2
    assert not sizing_config.is_optimization


@pytest.mark.unit
def test_the_shipped_optimization_case_parses(optimization_config):
    """Same aeroplane, five variables freed, four constraints, one objective."""
    assert len(optimization_config.design_variables) == 34
    assert len(optimization_config.free_variables) == 5
    assert len(optimization_config.constraints) == 4
    assert optimization_config.objective.name == "total_fuel"
    assert optimization_config.driver.name == "IPOPT"


@pytest.mark.unit
def test_every_design_variable_in_the_shipped_cases_records_a_source(sizing_config, optimization_config):
    """A number with no provenance is indistinguishable from a guess in the report quoting it."""
    for config in (sizing_config, optimization_config):
        unsourced = [name for name, spec in config.design_variables.items() if not spec.source]
        assert not unsourced, f"{config.path} has unsourced design variables: {unsourced}"


@pytest.mark.unit
def test_the_two_shipped_cases_describe_the_same_aeroplane(sizing_config, optimization_config):
    """The optimization case adds a question; it must not quietly change the aircraft."""
    sizing = {
        name: (np.asarray(spec.value).tolist(), spec.units) for name, spec in sizing_config.design_variables.items()
    }
    optimizing = {
        name: (np.asarray(spec.value).tolist(), spec.units)
        for name, spec in optimization_config.design_variables.items()
    }
    assert sizing == optimizing


@pytest.mark.unit
def test_the_shipped_cases_are_valid_yaml_documents():
    """Parsed independently of cdadt, so a YAML error is reported as one."""
    from tests.conftest import CASES

    for path in sorted(CASES.glob("*.yaml")):
        with open(path, encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), dict), path


@pytest.mark.unit
def test_a_section_carries_its_own_address_into_the_blocks_below_it():
    """The point of the class: a nested block derives its address rather than being told one."""
    section = CaseFileSection({"optimize": {"lower": 9.0, "upper": 11.0}}, "design_variables.ac|geom|wing|AR")

    assert section.child_of("optimize").where == "design_variables.ac|geom|wing|AR.optimize"
    # Unless it is a top-level block, which is addressed the way the file spells it.
    assert section.child_of("optimize", where="optimize").where == "optimize"


@pytest.mark.unit
def test_a_required_block_that_is_missing_names_the_section_that_wanted_it():
    """``ABSENT`` rather than ``None`` as the no-default marker, so ``None`` can be a default."""
    section = CaseFileSection({}, "the case file")

    with pytest.raises(ConfigError, match="'the case file' requires a 'black_box' key"):
        section.child_of("black_box")
    # Given a default, the same call is simply the default.
    assert section.child_of("solver", {}).where == "the case file.solver"


@pytest.mark.unit
def test_a_section_and_the_absent_marker_repr_as_what_they_are():
    """Both appear in tracebacks and in signatures, so both have to read as themselves."""
    section = CaseFileSection({"upper": 8000, "name": "takeoff_field_length"}, "constraints[0]")

    assert repr(section) == "CaseFileSection('constraints[0]', keys=['name', 'upper'])"
    assert repr(ABSENT) == "ABSENT"


@pytest.mark.unit
def test_every_scaling_spelling_is_readable_and_reaches_openmdao():
    """All four keys, not just ``ref``: the other three are what a case file may write instead.

    ``as_kwargs`` is what the driver is actually configured from, so reading the properties is
    not enough on its own -- the pair has to arrive.
    """
    assert Scaling(ref=2.0, ref0=0.5).ref0 == 0.5
    multiplicative = Scaling(scaler=4.0, adder=-1.0)
    assert (multiplicative.scaler, multiplicative.adder) == (4.0, -1.0)
    assert multiplicative.as_kwargs() == {"scaler": 4.0, "adder": -1.0}
    assert Scaling(ref=2.0, ref0=0.5).as_kwargs() == {"ref": 2.0, "ref0": 0.5}


@pytest.mark.unit
def test_a_design_variable_cannot_be_pinned_to_a_single_value():
    """``equals`` makes it a constraint, not a variable, and the driver would have nothing to move."""
    with pytest.raises(ConfigError, match="not by 'equals'"):
        OptimizeSpec(Bounds(equals=9.45))


@pytest.mark.unit
def test_a_bad_optimize_entry_is_reported_against_the_variable_it_belongs_to():
    """A case file cannot build a ``Bounds``, so the failure has to be re-raised with its address.

    Both spellings of a bad entry go through that same re-raise: an interval the wrong way round,
    caught by ``OptimizeSpec``, and two scalings at once, caught by ``Scaling``. An unknown *key*
    does not -- it is refused earlier, by name, and never reaches the constructor.
    """
    where = "ac|geom|wing|AR.optimize"
    with pytest.raises(ConfigError, match=r"'ac\|geom\|wing\|AR\.optimize': .*band is empty"):
        OptimizeSpec.from_section(CaseFileSection({"lower": 13.0, "upper": 7.0}, where))
    with pytest.raises(ConfigError, match=r"'ac\|geom\|wing\|AR\.optimize': .*not both"):
        OptimizeSpec.from_section(CaseFileSection({"lower": 7.0, "upper": 13.0, "ref": 1.0, "scaler": 2.0}, where))


@pytest.mark.unit
def test_a_vector_variable_can_be_freed_element_by_element():
    """A schedule may be optimized at some nodes and left alone at others."""
    spec = OptimizeSpec(Bounds(lower=0.0, upper=1.0), indices=[0, 2])

    assert spec.indices == [0, 2]
    assert spec.as_kwargs(units=None)["indices"] == [0, 2]
    # A copy, so a caller cannot reach in and re-free an element through the accessor.
    spec.indices.append(3)
    assert spec.indices == [0, 2]

    # And freeing all of them passes no 'indices' at all, rather than passing every index.
    everything = OptimizeSpec(Bounds(lower=0.0, upper=1.0))
    assert everything.indices is None
    assert "indices" not in everything.as_kwargs(units=None)


@pytest.mark.unit
def test_a_free_variable_reports_the_pieces_it_was_built_from():
    """The bounds and the scaling are read back by the report and the traceability matrix."""
    scaling = Scaling(ref=100.0)
    spec = OptimizeSpec(Bounds(lower=90.0, upper=180.0), scaling=scaling)

    assert spec.bounds.describe() == "90 to 180"
    assert spec.scaling is scaling


@pytest.mark.unit
def test_an_objective_may_be_one_element_of_a_vector_output():
    """Without an index OpenMDAO refuses a vector objective, so the key has to survive."""
    spec = ObjectiveSpec("fltcond|CL", index=3)

    assert spec.index == 3
    assert spec.as_kwargs(None)["index"] == 3
    assert ObjectiveSpec("total_fuel").index is None
    assert "index" not in ObjectiveSpec("total_fuel").as_kwargs(None)


@pytest.mark.unit
def test_maximizing_negates_the_whole_affine_map_and_not_only_its_slope():
    """``ref0`` shifts the objective, so leaving its sign alone would move the optimum."""
    spec = ObjectiveSpec("range", scaling=Scaling(ref=2.0, ref0=0.5), sense="maximize")
    arguments = spec.as_kwargs(None)

    assert (arguments["ref"], arguments["ref0"]) == (-2.0, -0.5)
    assert spec.scaling.ref == 2.0, "the spec itself is not mutated by being read"


@pytest.mark.unit
def test_a_bad_scaling_on_the_objective_is_reported_against_the_objective():
    """The two spellings are refused by ``Scaling``; the section has to be named in the message."""
    with pytest.raises(ConfigError, match="'objective': "):
        ObjectiveSpec.from_section(CaseFileSection({"name": "total_fuel", "ref": 1.0, "scaler": 2.0}, "objective"))


@pytest.mark.unit
def test_an_equality_constraint_reaches_openmdao_as_one():
    """``equals`` is a different keyword, not a degenerate pair of bounds."""
    assert Bounds(equals=0.024).as_kwargs() == {"equals": 0.024}


@pytest.mark.unit
def test_the_solver_iteration_limit_is_readable(sizing_config):
    """Read back when a study reports how the box it drove was configured."""
    assert sizing_config.solver.maxiter == sizing_config.solver.settings().maxiter


@pytest.mark.unit
def test_a_variable_knows_its_own_name(sizing_config):
    """The name is the key it was declared under, and the report is written from it."""
    for name, spec in sizing_config.design_variables.items():
        assert spec.name == name
