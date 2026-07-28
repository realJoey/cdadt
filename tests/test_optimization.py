"""Integration: the optimizer is driven by the case file, and checks it before it runs.

Two scales are exercised. A small, fast study proves the whole path -- case file to free
variables to constraints to a converged optimum -- and that changing the case file alone changes
what is optimized. The shipped study is then run in full, marked slow.
"""

from __future__ import annotations

import copy

import pytest

from cdadt import Config, Optimizer, SizingAnalysis
from cdadt.blackbox import BlackBoxError
from cdadt.optimization import OptimizationError
from tests.conftest import case_dict


def _case(free: dict | None = None, **sections) -> dict:
    """Return the shipped optimization case, shrunk to something that runs in seconds.

    Coarser grid, one free variable over a narrow interval, few iterations. Enough to prove the
    machinery drives the box; not a design study.
    """
    data = case_dict("b738_optimization.yaml")
    data["black_box"]["num_nodes"] = 5
    data["solver"]["maxiter"] = 60

    # Fix everything the shipped case frees, then free only what this test asks for.
    for spec in data["design_variables"].values():
        spec.pop("optimize", None)
    for name, bounds in (free or {"ac|geom|wing|AR": {"lower": 9.0, "upper": 10.5}}).items():
        target = data["design_variables"].get(name) or data["initial_conditions"][name]
        target["optimize"] = bounds

    data["driver"] = {"name": "IPOPT", "maxiter": 15, "tol": 1e-5, "derivative_mode": "fwd"}
    data["constraints"] = [
        {
            "name": "takeoff_field_length",
            "upper": 8000.0,
            "units": "ft",
            "regulation": "14 CFR 25.113",
            "source": "8000 ft dry runway at sea level, ISA",
        }
    ]
    data["objective"] = {"name": "total_fuel", "units": "kg", "ref": 2.0e4}
    data.update(copy.deepcopy(sections))
    return data


def _optimizer(**kwargs) -> Optimizer:
    """Return an optimizer over the small study."""
    return Optimizer(SizingAnalysis(Config.from_dict(_case(**kwargs))))


# =============================================================================================
# Checks that happen before anything expensive
# =============================================================================================


@pytest.mark.unit
def test_a_case_with_no_objective_cannot_be_optimized(sizing_config):
    """The message points at the class that does size without optimizing."""
    with pytest.raises(OptimizationError, match="declares no 'objective'"):
        Optimizer(SizingAnalysis(sizing_config))


@pytest.mark.integration
def test_a_misspelled_free_variable_is_caught_before_the_run():
    """Named, with suggestions, rather than thrown out of OpenMDAO's setup."""
    data = _case()
    data["design_variables"]["ac|geom|wing|Sref"] = {
        "value": 124.6,
        "units": "m**2",
        "source": "a typo",
        "optimize": {"lower": 90.0, "upper": 180.0},
    }
    with pytest.raises(BlackBoxError, match="did you mean"):
        Optimizer(SizingAnalysis(Config.from_dict(data)))
    # The correctly spelled case still builds, so the check is not rejecting everything.
    assert _optimizer() is not None


@pytest.mark.integration
def test_an_objective_the_box_does_not_publish_is_caught_before_the_run():
    """A constraint or objective on a quantity that does not exist fails early and by name."""
    with pytest.raises(OptimizationError, match="does not publish"):
        _optimizer(objective={"name": "specific_air_range"})


@pytest.mark.integration
def test_the_case_file_alone_decides_what_is_optimized():
    """Two studies of the same aeroplane differ only in data, which is the whole point."""
    fuel = _optimizer()
    weight = _optimizer(objective={"name": "MTOW", "units": "kg", "ref": 8.0e4}, constraints=[])

    assert fuel.objective_path == "mission.loiter.fuel_burn_integ.fuel_burn_final"
    assert weight.objective_path == "ac|weights|MTOW"
    assert fuel.objective_units == "kg" and weight.objective_units == "kg"
    assert len(fuel.basis) == 1 and len(weight.basis) == 0


@pytest.mark.integration
def test_a_mission_level_value_can_be_a_design_variable():
    """OpenConcept's B738_VLM_drag example optimizes ``cruise|h0``; so may a cdadt case.

    It lives in ``initial_conditions`` rather than ``design_variables``, and freeing it works
    the same way: an ``optimize:`` entry where the value is already declared.
    """
    optimizer = _optimizer(free={"cruise|h0": {"lower": 30000.0, "upper": 39000.0}})
    assert list(optimizer.analysis.config.free_variables) == ["cruise|h0"]
    # It resolves under the mission path, which is where the box actually publishes it.
    assert optimizer._resolved["cruise|h0"] == "mission.cruise|h0"


# =============================================================================================
# Running
# =============================================================================================


@pytest.mark.integration
def test_a_small_optimization_runs_end_to_end_and_improves_the_objective():
    """The whole path: case file, free variable, constraint, baseline, driver, optimum."""
    optimizer = _optimizer()
    outcome = optimizer.run()

    assert outcome.succeeded, [result.constraint.name for result in outcome.violated]
    assert outcome.optimum["total_fuel"] <= outcome.baseline["total_fuel"]
    # The design variable actually moved off its starting value, or nothing was optimized.
    # IPOPT holds its bounds to its own tolerance rather than exactly.
    aspect_ratio = optimizer.analysis.box.get("ac|geom|wing|AR")
    assert 9.0 - 1e-6 <= aspect_ratio <= 10.5 + 1e-6
    assert aspect_ratio != pytest.approx(9.45, abs=1e-9)


@pytest.mark.integration
def test_the_report_states_the_objective_the_variables_and_the_basis():
    """A report that omits what was constrained cannot be checked by a reader."""
    optimizer = _optimizer()
    report = optimizer.report(optimizer.run())

    assert "total_fuel" in report
    assert "ac|geom|wing|AR" in report
    assert "14 CFR 25.113" in report
    assert "8000 ft dry runway at sea level, ISA" in report
    assert "SUCCEEDED" in report


@pytest.mark.integration
def test_slsqp_is_the_default_driver_and_is_usable_on_a_narrow_study():
    """SciPy's SLSQP is always installed, and is the code's default for that reason.

    Exercised over a deliberately narrow interval, because the reason the shipped case uses
    IPOPT is that SLSQP cannot recover from a design the Newton solver fails to converge. Over a
    small interval no such design is proposed.
    """
    optimizer = _optimizer(
        free={"ac|geom|wing|AR": {"lower": 9.3, "upper": 9.7}},
        driver={"name": "SLSQP", "maxiter": 3, "tol": 1e-4, "derivative_mode": "fwd"},
        constraints=[],
    )
    outcome = optimizer.run()
    assert float(outcome.optimum["total_fuel"]) <= float(outcome.baseline["total_fuel"])
    assert len(optimizer.basis) == 0


@pytest.mark.integration
def test_pyoptsparse_settings_are_supplied_per_optimizer_and_can_be_overridden():
    """cdadt supplies defaults for IPOPT and SNOPT; the case file's own options win."""
    snopt = _optimizer(driver={"name": "SNOPT", "maxiter": 17, "tol": 1e-5})._pyoptsparse_settings()
    assert snopt["Major iterations limit"] == 17
    assert snopt["Major optimality tolerance"] == 1e-5

    ipopt = _optimizer(
        driver={"name": "IPOPT", "maxiter": 12, "options": {"print_level": 5, "mu_strategy": "monotone"}}
    )._pyoptsparse_settings()
    assert ipopt["max_iter"] == 12
    assert ipopt["hessian_approximation"] == "limited-memory"
    assert ipopt["print_level"] == 5, "the case file's own option must win over cdadt's default"
    assert ipopt["mu_strategy"] == "monotone"


@pytest.mark.integration
def test_a_driver_that_is_not_pyoptsparse_supplies_no_pyoptsparse_settings():
    """SLSQP is a SciPy driver; there are no pyOptSparse options to hand it."""
    assert _optimizer(driver={"name": "SLSQP", "maxiter": 5})._pyoptsparse_settings() == {}


@pytest.mark.integration
def test_the_report_says_plainly_when_a_study_did_not_succeed():
    """Both failure modes must read as failures: a violated constraint, and a failed driver.

    Built from a real converged optimizer, so the report has a built box to read variables from,
    then handed outcomes that did not succeed. A report printing "SUCCEEDED" over a violated
    basis would be wrong exactly where a reader looks.
    """
    from cdadt.config import Bounds
    from cdadt.optimization import OptimizationOutcome

    optimizer = _optimizer()
    outcome = optimizer.run()
    assert "SUCCEEDED" in optimizer.report(outcome)

    # Re-evaluate the same basis against an impossible bound, so every constraint is violated.
    for constraint in optimizer.basis:
        constraint.spec._bounds = Bounds(upper=-1.0)
    violated = optimizer.basis.evaluate(optimizer.analysis.box)
    assert violated and not any(result.satisfied for result in violated)

    basis_failed = OptimizationOutcome(
        outcome.baseline, outcome.optimum, violated, outcome.objective, outcome.sense, failed=False
    )
    assert not basis_failed.succeeded
    assert "DID NOT SUCCEED: violated" in optimizer.report(basis_failed)

    driver_failed = OptimizationOutcome(
        outcome.baseline, outcome.optimum, [], outcome.objective, outcome.sense, failed=True
    )
    assert not driver_failed.succeeded
    assert "DID NOT SUCCEED: the driver reported failure" in optimizer.report(driver_failed)


@pytest.mark.integration
def test_the_outcome_and_optimizer_repr_as_what_they_describe():
    """Both are what a debugger frame shows while a study is being diagnosed."""
    optimizer = _optimizer()
    outcome = optimizer.run()

    assert repr(optimizer).startswith("Optimizer(objective='total_fuel'")
    assert repr(outcome) == "OptimizationOutcome('total_fuel', succeeded=True)"
    assert all(result.active for result in outcome.active)
    assert not outcome.violated
    assert [result.constraint.name for result in outcome.constraints] == ["takeoff_field_length"]


@pytest.mark.integration
@pytest.mark.slow
def test_the_shipped_optimization_converges_and_meets_every_constraint(optimization_config):
    """The study the repository advertises, run in full."""
    optimizer = Optimizer(SizingAnalysis(optimization_config))
    outcome = optimizer.run()

    assert outcome.succeeded, [result.constraint.name for result in outcome.violated]
    assert not outcome.violated
    # The optimization must actually have found something, not merely have terminated.
    assert outcome.optimum["total_fuel"] < outcome.baseline["total_fuel"] * 0.99
    # And it must still be flying the mission it was asked to fly.
    assert outcome.optimum["mission_range_flown"] == pytest.approx(2800.0, rel=1e-6)
    assert outcome.optimum["takeoff_field_length"] == pytest.approx(outcome.optimum["abort_distance"], rel=1e-6)
