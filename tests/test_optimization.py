"""Integration: the optimizer is driven by the case file, and checks it before it runs.

Two scales are exercised. A small, fast optimization proves the whole path -- case file to
design variables to constraints to a converged optimum -- and that changing the case file alone
changes what is optimized. The shipped study is then run in full, marked slow, and its
requirements checked at the optimum.
"""

from __future__ import annotations

import copy

import pytest

from cdadt import Config, Optimizer, SizingAnalysis
from cdadt.blackbox import BlackBoxError
from cdadt.optimization import OptimizationError


def _small(optimization_config: Config, **changes) -> Config:
    """Return the shipped optimization case, shrunk to something that runs in seconds.

    Coarser grid, one design variable over a narrow interval, few iterations. Enough to prove
    the machinery drives the box; not a design study.
    """
    data = _as_dict(optimization_config)
    data["black_box"]["num_nodes"] = 5
    data["optimization"] = {
        "driver": {"name": "IPOPT", "maxiter": 15, "tol": 1e-5, "derivative_mode": "fwd"},
        "objective": {"name": "total_fuel", "units": "kg", "ref": 2.0e4},
        "design_variables": [{"name": "ac|geom|wing|AR", "lower": 9.0, "upper": 10.5}],
        "requirements": [
            {
                "type": "balanced_field_length",
                "limit": 8000.0,
                "units": "ft",
                "regulation": "14 CFR 25.113",
                "source": "8000 ft dry runway at sea level, ISA",
            }
        ],
    }
    for section, override in changes.items():
        data[section] = override
    return Config.from_dict(data)


def _as_dict(config: Config) -> dict:
    """Re-read a shipped case file as a plain dictionary, so tests can vary it."""
    import yaml

    with open(config.path, encoding="utf-8") as handle:
        return copy.deepcopy(yaml.safe_load(handle))


# =============================================================================================
# Checks that happen before anything expensive
# =============================================================================================


@pytest.mark.unit
def test_a_case_with_no_optimization_section_cannot_be_optimized(sizing_config):
    """The message points at the class that does size without optimizing."""
    with pytest.raises(OptimizationError, match="declares no 'optimization' section"):
        Optimizer(SizingAnalysis(sizing_config))


@pytest.mark.integration
def test_a_misspelled_design_variable_is_caught_before_the_run(optimization_config):
    """Named, with suggestions, rather than thrown out of OpenMDAO's setup."""
    config = _small(optimization_config)
    data = _as_dict(optimization_config)
    data["black_box"]["num_nodes"] = 5
    data["optimization"] = {
        "objective": {"name": "total_fuel"},
        "design_variables": [{"name": "ac|geom|wing|Sref", "lower": 90, "upper": 180, "units": "m**2"}],
    }
    with pytest.raises(BlackBoxError, match="did you mean"):
        Optimizer(SizingAnalysis(Config.from_dict(data)))
    # The correctly spelled case still builds, so the check is not simply rejecting everything.
    assert Optimizer(SizingAnalysis(config)) is not None


@pytest.mark.integration
def test_an_objective_the_box_does_not_publish_is_caught_before_the_run(optimization_config):
    """A constraint or objective on a quantity that does not exist fails early and by name."""
    data = _as_dict(optimization_config)
    data["black_box"]["num_nodes"] = 5
    data["optimization"] = {
        "objective": {"name": "specific_air_range"},
        "design_variables": [{"name": "ac|geom|wing|AR", "lower": 9.0, "upper": 10.5}],
    }
    with pytest.raises(OptimizationError, match="does not publish"):
        Optimizer(SizingAnalysis(Config.from_dict(data)))


@pytest.mark.integration
def test_the_case_file_alone_decides_what_is_optimized(optimization_config):
    """Two studies of the same aeroplane differ only in data, which is the whole point."""
    fuel = Optimizer(SizingAnalysis(_small(optimization_config)))
    weight_case = _as_dict(optimization_config)
    weight_case["black_box"]["num_nodes"] = 5
    weight_case["optimization"] = {
        "objective": {"name": "MTOW", "units": "kg", "ref": 8.0e4},
        "design_variables": [{"name": "ac|geom|wing|AR", "lower": 9.0, "upper": 10.5}],
    }
    weight = Optimizer(SizingAnalysis(Config.from_dict(weight_case)))

    assert fuel.objective_path == "mission.loiter.fuel_burn_integ.fuel_burn_final"
    assert weight.objective_path == "ac|weights|MTOW"
    assert fuel.objective_units == "kg" and weight.objective_units == "kg"
    assert len(fuel.basis) == 1 and len(weight.basis) == 0


# =============================================================================================
# Running
# =============================================================================================


@pytest.mark.integration
def test_a_small_optimization_runs_end_to_end_and_improves_the_objective(optimization_config):
    """The whole path: case file, design variable, constraint, baseline, driver, optimum."""
    optimizer = Optimizer(SizingAnalysis(_small(optimization_config)))
    outcome = optimizer.run()

    assert outcome.succeeded, [result.requirement.name for result in outcome.violated]
    assert outcome.optimum["total_fuel"] <= outcome.baseline["total_fuel"]
    # The design variable actually moved off its starting value, or nothing was optimized.
    # IPOPT holds its bounds to its own small tolerance rather than exactly, so the check
    # allows for that rather than asserting a strict inequality it need not satisfy.
    bound_tolerance = 1e-6
    aspect_ratio = optimizer.analysis.box.get("ac|geom|wing|AR")
    assert 9.0 - bound_tolerance <= aspect_ratio <= 10.5 + bound_tolerance
    assert aspect_ratio != pytest.approx(9.45, abs=1e-9)


@pytest.mark.integration
def test_the_report_states_the_objective_the_design_variables_and_the_basis(optimization_config):
    """A report that omits what was constrained cannot be checked by a reader."""
    optimizer = Optimizer(SizingAnalysis(_small(optimization_config)))
    report = optimizer.report(optimizer.run())

    assert "total_fuel" in report
    assert "ac|geom|wing|AR" in report
    assert "14 CFR 25.113" in report
    assert "8000 ft dry runway at sea level, ISA" in report
    assert "SUCCEEDED" in report


@pytest.mark.integration
@pytest.mark.slow
def test_the_shipped_optimization_converges_and_meets_every_requirement(optimization_config):
    """The study the repository advertises, run in full."""
    optimizer = Optimizer(SizingAnalysis(optimization_config))
    outcome = optimizer.run()

    assert outcome.succeeded, [result.requirement.name for result in outcome.violated]
    assert not outcome.violated
    # The optimization must actually have found something, not merely have terminated.
    assert outcome.optimum["total_fuel"] < outcome.baseline["total_fuel"] * 0.99
    # And it must still be flying the mission it was asked to fly.
    assert outcome.optimum["mission_range_flown"] == pytest.approx(2800.0, rel=1e-6)
    assert outcome.optimum["takeoff_field_length"] == pytest.approx(outcome.optimum["abort_distance"], rel=1e-6)
