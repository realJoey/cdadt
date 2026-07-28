"""Verification: derivatives. Are the gradients the optimizer steps on correct?

Every design change cdadt reports comes from an optimizer following total derivatives taken
across the whole Newton-converged coupled system -- the weight closure, every phase duration,
the throttle solves and the implicit decision speed, all at once. If those derivatives are
wrong, the optimizer still converges: it converges to the wrong design, confidently, and
nothing in a results table shows it.

So they are checked against finite differences of the converged model, at three step sizes.
Three, rather than one, because a single step size cannot distinguish a correct derivative from
a coincidence: a correct analytic derivative shows the classic V -- truncation error falling as
the step shrinks, round-off rising as it shrinks further, and a minimum in between. Seeing that
shape is the evidence; the single best number is only its bottom.

The floor on how well FD can agree here is set by the solver: the converged state itself is
only good to about 1e-9, so differencing it over a relative step of 1e-6 cannot do better than
roughly 1e-4. Agreement at that level is the correct expectation, and asking for 1e-10 would be
asking the finite difference to be more accurate than the thing it differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdadt import Config, SizingAnalysis
from cdadt.optimization import Optimizer

#: Relative finite-difference steps to sweep. The middle one should be the best.
STEPS: tuple[float, ...] = (1e-5, 1e-6, 1e-7)

#: The best step must achieve at least this agreement on every non-negligible derivative.
REQUIRED_AGREEMENT = 1e-3

#: Derivatives smaller than this in magnitude are excluded: a relative error on a derivative
#: that is numerically zero is a division by noise, not a measurement.
NEGLIGIBLE = 1e-8


#: The three variables freed for the check, as ``name -> optimize entry``. A variable is freed
#: by gaining an ``optimize:`` entry where the case file already declares it, so the units and
#: the value come from that declaration and are not restated here.
FREED = {
    "ac|geom|wing|S_ref": {"lower": 90.0, "upper": 180.0},
    "ac|geom|wing|AR": {"lower": 7.0, "upper": 13.0},
    "ac|propulsion|engine|rating": {"lower": 18.0e3, "upper": 34.0e3},
}


@pytest.fixture(scope="module")
def derivative_check(optimization_case):
    """Build, converge, and check totals at every step size.

    Runs on the optimization case with the vector throttle constraints removed, leaving scalar
    responses. That is not a weaker check -- it is the same derivative machinery -- and it keeps
    the comparison table readable.
    """
    case = optimization_case()
    case["black_box"]["num_nodes"] = 11
    case["solver"].update({"maxiter": 60, "atol": 1e-9, "rtol": 1e-9})
    for name, spec in case["design_variables"].items():
        spec.pop("optimize", None)
        if name in FREED:
            spec["optimize"] = dict(FREED[name])
    case["constraints"] = [
        constraint for constraint in case["constraints"] if not constraint["name"].endswith("_throttle")
    ]

    optimizer = Optimizer(SizingAnalysis(Config.from_dict(case)))
    optimizer.prepare(driver=False)
    problem = optimizer.analysis.box.problem

    sweep = {}
    for step in STEPS:
        checked = problem.check_totals(method="fd", step=step, step_calc="rel_avg", out_stream=None)
        sweep[step] = {
            (of, wrt): (
                float(np.atleast_1d(data["J_fwd"]).ravel()[0]),
                float(np.atleast_1d(data["J_fd"]).ravel()[0]),
            )
            for (of, wrt), data in checked.items()
        }
    return sweep


def _relative_errors(pairs: dict) -> dict:
    """Return the relative error of each non-negligible derivative."""
    return {
        key: abs(analytic - finite) / abs(finite)
        for key, (analytic, finite) in pairs.items()
        if abs(finite) > NEGLIGIBLE
    }


@pytest.mark.verification
@pytest.mark.slow
def test_there_are_derivatives_to_check(derivative_check):
    """Guard against a vacuous pass: a check over an empty set proves nothing."""
    pairs = derivative_check[1e-6]
    assert len(pairs) >= 9, f"expected at least 3 responses x 3 design variables, got {len(pairs)}"
    assert _relative_errors(pairs), "every derivative was treated as negligible"


@pytest.mark.verification
@pytest.mark.slow
def test_analytic_totals_agree_with_finite_differences(derivative_check):
    """At the best step size, every meaningful derivative must agree to the stated tolerance."""
    errors = _relative_errors(derivative_check[1e-6])
    bad = {key: error for key, error in errors.items() if error > REQUIRED_AGREEMENT}
    assert not bad, "Total derivatives disagree with finite differences:\n  " + "\n  ".join(
        f"{of} wrt {wrt}: rel {error:.2e}" for (of, wrt), error in bad.items()
    )


@pytest.mark.verification
@pytest.mark.slow
def test_the_step_size_sweep_shows_the_expected_truncation_and_round_off_behaviour(derivative_check):
    """The middle step must be the best of the three.

    This is the check that distinguishes a correct derivative from a lucky one. A wrong analytic
    derivative differs from FD by a roughly constant amount at every step, so its error curve is
    flat. A correct one is limited by truncation at large steps and by round-off at small ones,
    so its error has a minimum in between.
    """
    worst = {step: max(_relative_errors(pairs).values()) for step, pairs in derivative_check.items()}
    assert worst[1e-6] < worst[1e-5], f"no truncation-error reduction from 1e-5 to 1e-6: {worst}"
    assert worst[1e-6] < worst[1e-7], f"no round-off growth from 1e-6 to 1e-7: {worst}"


@pytest.mark.verification
@pytest.mark.slow
def test_the_signs_of_the_important_derivatives_are_physically_right(derivative_check):
    """Sign errors are the failure mode that sends an optimizer confidently the wrong way.

    Three that can be reasoned about without any model:

    - More wing span at fixed area (higher aspect ratio) cuts induced drag, so fuel falls.
    - More aspect ratio improves the engine-out climb gradient, for the same reason.
    - More installed thrust shortens the balanced field length.
    """
    pairs = derivative_check[1e-6]

    def derivative(response_contains: str, variable_contains: str) -> float:
        matches = [
            analytic
            for (of, wrt), (analytic, _) in pairs.items()
            if response_contains in of and variable_contains in wrt
        ]
        assert len(matches) == 1, f"expected exactly one derivative of {response_contains} wrt {variable_contains}"
        return matches[0]

    assert derivative("fuel_burn_final", "AR") < 0.0, "raising aspect ratio should cut fuel"
    assert derivative("engine_out_climb_gradient", "AR") > 0.0, "raising aspect ratio should improve the gradient"
    assert derivative("takeoff_field_length", "rating") < 0.0, "more thrust should shorten the field"
