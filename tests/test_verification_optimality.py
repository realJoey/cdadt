"""Verification: optimality. Is the reported optimum actually an optimum?

A driver reporting success means it satisfied its own termination criteria. That is not the same
as having found a constrained minimum, and the difference is invisible in a results table: a run
that stalled on its iteration limit, or converged to a point where a constraint gradient was
mis-signed, prints exactly like a converged optimum.

So the optimum is probed directly. Around a genuine constrained minimum, every feasible
direction must either fail to improve the objective or leave the feasible set. This module
perturbs each design variable in both directions and checks precisely that.

It runs on a deliberately small study -- one design variable, a narrow interval, one
requirement -- because the property being tested is a property of the optimizer and the
derivative chain, not of the design space, and testing it cheaply means it can be tested at all.
"""

from __future__ import annotations

import pytest

from cdadt import Config, SizingAnalysis
from cdadt.optimization import Optimizer

#: Relative perturbation applied to each design variable. Large enough to move the objective
#: well clear of solver noise, small enough to stay in the neighbourhood of the optimum.
PERTURBATION = 0.02

#: The objective is allowed to improve by at most this much under perturbation before the point
#: is judged not to be an optimum. Sized above the solver's own noise floor.
IMPROVEMENT_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def optimized(optimization_case):
    """Run a small optimization to convergence and return the optimizer and its outcome."""
    case = optimization_case()
    case["black_box"]["num_nodes"] = 5
    case["solver"].update({"maxiter": 60})
    case["optimization"]["driver"] = {
        "name": "IPOPT",
        "maxiter": 25,
        "tol": 1e-7,
        "derivative_mode": "fwd",
    }
    case["optimization"]["design_variables"] = [{"name": "ac|geom|wing|AR", "lower": 8.5, "upper": 10.0}]
    case["optimization"]["requirements"] = [
        requirement
        for requirement in case["optimization"]["requirements"]
        if requirement["type"] == "balanced_field_length"
    ]

    optimizer = Optimizer(SizingAnalysis(Config.from_dict(case)))
    outcome = optimizer.run()
    return optimizer, outcome


@pytest.mark.verification
@pytest.mark.slow
def test_the_driver_reported_success_and_the_basis_holds(optimized):
    """Both halves of what cdadt calls success, checked before anything is inferred from it."""
    _optimizer, outcome = optimized
    assert outcome.succeeded, [result.requirement.name for result in outcome.violated]


@pytest.mark.verification
@pytest.mark.slow
def test_the_optimum_is_at_least_as_good_as_the_baseline(optimized):
    """A necessary condition, and one that has caught real sign errors elsewhere."""
    _optimizer, outcome = optimized
    assert float(outcome.optimum["total_fuel"]) <= float(outcome.baseline["total_fuel"])


@pytest.mark.verification
@pytest.mark.slow
def test_no_feasible_perturbation_of_a_design_variable_improves_the_objective(optimized):
    """The defining property of a constrained optimum, tested by direct probing.

    For each design variable, step it up and down, reconverge the black box at the perturbed
    design, and require that the objective either got worse, or that the step left the variable
    outside its own bounds, or that it violated a requirement. An improving feasible direction
    would mean the reported optimum is not one.
    """
    optimizer, outcome = optimized
    box = optimizer.analysis.box
    catalog = optimizer.analysis.catalog
    objective_path = optimizer.objective_path
    objective_units = optimizer.objective_units

    optimum_objective = float(box.get(objective_path, units=objective_units))
    improvements = []

    for variable in optimizer.settings.design_variables:
        at_optimum = float(box.get(variable.name, units=variable.units))
        for direction in (+1.0, -1.0):
            probe = at_optimum * (1.0 + direction * PERTURBATION)
            if not (variable.lower <= probe <= variable.upper):
                continue  # the step leaves the design space; not a feasible direction

            box.set(variable.name, probe, units=variable.units)
            box.run()
            probed_objective = float(box.get(objective_path, units=objective_units))
            feasible = all(requirement.evaluate(box, catalog).satisfied for requirement in optimizer.basis)
            improvement = (optimum_objective - probed_objective) / abs(optimum_objective)
            if feasible and improvement > IMPROVEMENT_TOLERANCE:
                improvements.append(
                    f"{variable.name} {'+' if direction > 0 else '-'}{PERTURBATION:.0%}: "
                    f"objective improved by {improvement:.2e} while staying feasible"
                )

            box.set(variable.name, at_optimum, units=variable.units)

        box.run()  # restore the converged optimum before the next variable

    assert not improvements, "The reported optimum is not a local optimum:\n  " + "\n  ".join(improvements)
    assert outcome.succeeded


@pytest.mark.verification
@pytest.mark.slow
def test_the_active_set_is_reported_honestly(optimized):
    """A requirement called active must really be on its limit, and one called met must not be.

    The traceability matrix is the artefact a certification argument is built from. If it
    labelled a slack requirement active, or a binding one merely met, the design story it tells
    would be wrong in the one place a reader looks.
    """
    _optimizer, outcome = optimized
    for result in outcome.requirements:
        if result.active:
            assert abs(result.relative_margin) <= result.ACTIVE_TOLERANCE, (
                f"{result.requirement.name} is reported ACTIVE with relative margin " f"{result.relative_margin:.2e}"
            )
        elif result.satisfied:
            assert (
                result.relative_margin > result.ACTIVE_TOLERANCE
            ), f"{result.requirement.name} is reported MET but sits on its limit"
