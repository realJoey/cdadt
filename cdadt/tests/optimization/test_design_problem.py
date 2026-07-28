"""Tests for the design optimization layer.

Claim classes: ``unit`` for the problem definition, ``integration``/``regression`` for the
converged optimization.

The test that matters most here asserts the optimizer reported **Optimal**. An OpenMDAO
``run_driver`` returns whether it finished, not whether it succeeded: a run that hit its
iteration limit or failed to restore feasibility still yields a design vector and a full set
of results, and those are indistinguishable from an optimum unless something checks. So the
integration test asserts the exit status, not merely that the objective went down.
"""

from __future__ import annotations

import pytest

from cdadt.aircraft import jet_transport_disciplines
from cdadt.certification import (
    ApproachSpeedLimit,
    BalancedFieldLength,
    CertificationBasis,
    EngineOutClimbGradient,
    LandingFieldLengthLimit,
    ThrottleMargin,
)
from cdadt.core.configuration import MissingConfigurationError
from cdadt.optimization import DesignProblem, DesignVariable, IpoptDriver, SlsqpDriver

ENGINE_DECK = "CFM56"
THROTTLED_PHASES = ("climb", "cruise", "descent", "reserve_climb", "reserve_cruise", "loiter")


def _basis(config):
    """Return the full B738 certification basis."""
    return CertificationBasis(
        [
            BalancedFieldLength(config),
            EngineOutClimbGradient(config),
            LandingFieldLengthLimit(config),
            ApproachSpeedLimit(config),
            *[ThrottleMargin(config, phase=phase) for phase in THROTTLED_PHASES],
        ]
    )


def _problem(config, objective="total_fuel", driver=None, num_nodes=11):
    """Return an assembled design problem for the B738 case."""
    return DesignProblem(
        disciplines=jet_transport_disciplines(config, engine_deck=ENGINE_DECK),
        config=config,
        certification=_basis(config),
        driver=driver or IpoptDriver(config),
        objective=objective,
        num_nodes=num_nodes,
    )


# ==============================================================================
# Design variables
# ==============================================================================
@pytest.mark.unit
def test_design_variables_are_read_from_configuration(b738_config):
    """Every configured design variable appears with its bounds and units."""
    names = {variable.name for variable in _problem(b738_config).design_variables}
    assert names == {
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|taper",
        "ac|propulsion|engine|rating",
    }


@pytest.mark.unit
def test_a_design_variable_with_an_empty_range_is_rejected():
    """Bounds that cross make the variable silently fixed, so they are refused."""
    with pytest.raises(ValueError, match="at or above its upper"):
        DesignVariable(name="ac|geom|wing|AR", lower=11.0, upper=9.0, units=None)


@pytest.mark.unit
def test_design_variables_are_non_dimensionalized_by_default():
    """An unconfigured scaler makes the variable order one over its range.

    Aspect ratio spans 7 to 11 and engine rating 20,000 to 34,000 lbf. Presented unscaled,
    an optimizer spends its progress on the large number.
    """
    aspect_ratio = DesignVariable("ac|geom|wing|AR", lower=7.0, upper=11.0, units=None)
    rating = DesignVariable("ac|propulsion|engine|rating", lower=20e3, upper=34e3, units="lbf")

    assert aspect_ratio.effective_scaler == pytest.approx(1.0 / 11.0)
    assert rating.effective_scaler == pytest.approx(1.0 / 34e3)


@pytest.mark.unit
def test_a_configured_scaler_wins_over_the_default():
    """A particular problem may need different treatment, so configuration overrides."""
    variable = DesignVariable("ac|geom|wing|AR", lower=7.0, upper=11.0, units=None, scaler=0.5)
    assert variable.effective_scaler == 0.5


@pytest.mark.unit
def test_an_optimization_with_no_design_variables_is_rejected(b738_config):
    """Nothing to vary means the baseline gets reported as an optimum."""
    stripped = b738_config.subtree("ac")
    # Rebuild a config that has everything except the design variable declarations.
    without_dvs = b738_config.as_dict()
    del without_dvs["optimization"]["design_variables"]
    from cdadt.core.configuration import AircraftConfiguration

    assert stripped is not None
    with pytest.raises(ValueError, match="No design variables are configured"):
        _problem(AircraftConfiguration(without_dvs))


# ==============================================================================
# The objective
# ==============================================================================
@pytest.mark.unit
def test_the_objective_must_be_something_the_model_produces(b738_config):
    """A misspelled objective fails at construction, listing what is available."""
    with pytest.raises(ValueError, match="not a quantity this model produces"):
        _problem(b738_config, objective="blockfuel")


@pytest.mark.unit
@pytest.mark.parametrize("objective", ["total_fuel", "block_fuel", "MTOW", "OEW"])
def test_supported_objectives_resolve_to_a_path(b738_config, objective):
    """Each supported objective maps to a real problem path."""
    assert _problem(b738_config, objective=objective).objective_path()


@pytest.mark.unit
def test_the_objective_has_no_default(b738_config):
    """What a design is optimized for must be stated, never inherited."""
    import inspect

    parameter = inspect.signature(DesignProblem.__init__).parameters["objective"]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.unit
def test_the_objective_reference_is_required(b738_config):
    """Scaling the objective needs a magnitude, and there is no sensible default.

    A fuel burn and an empty weight are both in kilograms and differ by a factor of three;
    a takeoff distance in feet by three more.
    """
    from cdadt.core.configuration import AircraftConfiguration

    data = b738_config.as_dict()
    del data["optimization"]["objective_reference"]
    problem = _problem(AircraftConfiguration(data))
    with pytest.raises(MissingConfigurationError, match="objective_reference"):
        problem.build()


# ==============================================================================
# Drivers
# ==============================================================================
@pytest.mark.unit
def test_drivers_require_their_settings(b738_config):
    """An optimizer with no iteration limit or tolerance refuses to be built."""
    stripped = b738_config.subtree("ac")
    for driver_class in (IpoptDriver, SlsqpDriver):
        with pytest.raises(MissingConfigurationError):
            driver_class(stripped)


@pytest.mark.unit
def test_a_driver_that_never_ran_does_not_report_an_optimum(b738_config):
    """Asking for the outcome before a run reports failure, not success.

    The default must be "not optimal". A status object that defaulted to success would let
    a run that never reached the optimizer be reported as converged.
    """
    for driver_class in (IpoptDriver, SlsqpDriver):
        driver = driver_class(b738_config)
        outcome = driver.outcome(driver.build())
        assert not outcome.optimal
        assert "did not" in outcome.status


@pytest.mark.unit
def test_ipopt_treats_only_solve_succeeded_as_optimal(b738_config):
    """IPOPT's "solved to acceptable level" is not reported as an optimum.

    Acceptable means IPOPT relaxed its own tolerances because it stopped making progress. A
    design study should say so rather than present the point as converged.
    """
    assert IpoptDriver.SOLVED == 0


# ==============================================================================
# The converged optimization
# ==============================================================================
@pytest.fixture(scope="module")
def optimized_b738(b738_config):
    """Run the B738 fuel-burn optimization with IPOPT and return the result."""
    design_problem = _problem(b738_config)
    problem = design_problem.build()
    result = design_problem.run(problem)
    return design_problem, result


@pytest.mark.integration
@pytest.mark.slow
def test_ipopt_reaches_an_optimum(optimized_b738):
    """IPOPT reports Solve Succeeded, not merely that it stopped.

    This is the claim the whole optimization layer rests on. Without it, an iteration-limit
    exit produces a full set of plausible results that nothing distinguishes from an
    optimum.
    """
    _, result = optimized_b738
    assert result.optimal, f"IPOPT did not reach an optimum: {result.outcome.status}"
    assert "Solve Succeeded" in result.outcome.status


@pytest.mark.integration
@pytest.mark.slow
def test_the_optimum_improves_on_the_baseline(optimized_b738):
    """The optimized design burns less fuel than the baseline it started from.

    A converged optimizer that improved nothing usually means the design variables never
    reached the model.
    """
    _, result = optimized_b738
    assert result.objective_value < result.baseline_objective, (
        f"Optimized {result.objective_name} of {result.objective_value:.1f} is not better than the "
        f"baseline {result.baseline_objective:.1f}."
    )
    assert (
        result.improvement > 0.01
    ), f"Only {result.improvement:.4%} improvement; check the design variables reach the model."


@pytest.mark.integration
@pytest.mark.slow
def test_every_certification_requirement_is_satisfied_at_the_optimum(optimized_b738):
    """No requirement is violated at the reported optimum.

    An optimizer can report success on a point that is optimal but infeasible if the
    constraints were never registered, so this reads the requirements back independently of
    what the optimizer said.
    """
    _, result = optimized_b738
    assert "0 not met" in result.traceability_matrix


@pytest.mark.integration
@pytest.mark.slow
def test_the_optimum_lies_within_the_configured_bounds(optimized_b738):
    """Every design variable is inside its bounds, which keep the methods valid.

    The empirical weight and drag buildups were fitted over a range; outside it they still
    return numbers.
    """
    design_problem, result = optimized_b738
    for variable in design_problem.design_variables:
        value = result.design_variables[variable.name]
        assert (
            variable.lower - 1e-6 <= value <= variable.upper + 1e-6
        ), f"{variable.name} = {value} is outside [{variable.lower}, {variable.upper}]"


@pytest.mark.integration
@pytest.mark.slow
def test_at_least_one_requirement_is_active_at_the_optimum(optimized_b738):
    """Some requirement binds, which is what makes this certification-*driven*.

    If no requirement were active, the certification basis did not shape the design and the
    result is just an unconstrained minimum that happens to be legal.
    """
    _, result = optimized_b738
    assert "active" in result.traceability_matrix
    assert "Active requirements shaped this design" in result.traceability_matrix


@pytest.mark.integration
@pytest.mark.slow
def test_the_report_leads_with_whether_the_result_is_an_optimum(optimized_b738):
    """The exit status appears before any numbers.

    A reader who skims the design variables and never sees that the run stopped short has
    been misled by the report's layout rather than by its contents.
    """
    _, result = optimized_b738
    report = result.report()
    assert report.index("optimal:") < report.index("Design variables")


@pytest.mark.regression
@pytest.mark.slow
def test_the_optimum_has_not_drifted(optimized_b738):
    """Pin the converged optimum.

    A *regression* claim: the answer has not changed, not that it was ever right. See
    ``cdadt/tests/validation/test_b738_sizing.py`` for what can and cannot currently be
    validated externally.
    """
    _, result = optimized_b738
    golden = {
        "ac|geom|wing|S_ref": 131.49,
        "ac|geom|wing|AR": 11.0,
        "ac|geom|wing|c4sweep": 15.0,
        "ac|geom|wing|taper": 0.2467,
        "ac|propulsion|engine|rating": 22184.8,
    }
    drifted = {
        name: (expected, result.design_variables[name])
        for name, expected in golden.items()
        if result.design_variables[name] != pytest.approx(expected, rel=1e-3)
    }
    assert not drifted, f"Optimum has drifted (expected, actual): {drifted}"
    assert result.objective_value == pytest.approx(17164.3, rel=1e-3)
