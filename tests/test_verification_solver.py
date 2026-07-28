"""Verification: the solver. Is the coupled system actually converged, and to what?

Three questions, each of which a sizing tool can get wrong silently.

**Does a failed solve fail loudly?** A non-converged mission still produces numbers -- a
negative field length, a range that misses the one requested -- and nothing about them says so.
The case file's ``err_on_non_converge`` must therefore really raise.

**What residual can the box actually reach?** Measured, not assumed. It turns out to be about
1e-9 regardless of grid, which is an absolute floor of the box rather than something that grows
with problem size. That single measurement explains the shipped tolerance, and explains why the
grid study in :mod:`tests.test_verification_grid` runs at 1e-8.

**Is the converged answer independent of how tightly it was converged?** If tightening the
tolerance moves the answer, the answer was never converged.
"""

from __future__ import annotations

import pytest
from openmdao.core.analysis_error import AnalysisError

from cdadt import Config, SizingAnalysis

#: Tolerances to probe for the achievable residual floor.
PROBED_TOLERANCES: tuple[float, ...] = (1e-7, 1e-8, 1e-9, 1e-10)

#: The tolerance the shipped cases request.
SHIPPED_TOLERANCE = 1e-9


def _converges(case: dict, num_nodes: int, tolerance: float, maxiter: int = 80) -> bool:
    """Return whether the case converges at the given grid and tolerance."""
    case["black_box"]["num_nodes"] = num_nodes
    case.setdefault("solver", {}).update({"maxiter": maxiter, "atol": tolerance, "rtol": tolerance})
    try:
        SizingAnalysis(Config.from_dict(case)).run()
    except AnalysisError:
        return False
    return True


@pytest.mark.verification
def test_a_failed_solve_raises_rather_than_returning_numbers(sizing_case):
    """With one Newton iteration allowed, the design mission cannot converge, and must say so.

    The most important negative test in the repository. Everything else assumes that a result
    which came back is a result that converged.
    """
    case = sizing_case()
    case["solver"] = {"maxiter": 1, "atol": 1e-12, "rtol": 1e-12, "err_on_non_converge": True}
    with pytest.raises(AnalysisError):
        SizingAnalysis(Config.from_dict(case)).run()


@pytest.mark.verification
def test_a_failed_solve_can_be_allowed_through_only_by_asking_for_it(sizing_case):
    """``err_on_non_converge: false`` returns the state reached, which is a deliberate choice.

    Supported because a study may want to inspect a failure, and dangerous because the result
    is indistinguishable from a converged one. The case file has to say so explicitly; there is
    no default that quietly permits it.
    """
    case = sizing_case()
    case["solver"] = {"maxiter": 1, "atol": 1e-12, "rtol": 1e-12, "err_on_non_converge": False}
    results = SizingAnalysis(Config.from_dict(case)).run()
    # It returns something. That something is not a converged aircraft, and the balanced field
    # length is the quantity that shows it: V1 has not been solved.
    assert results["takeoff_field_length"] != pytest.approx(results["abort_distance"], rel=1e-6)


@pytest.mark.verification
@pytest.mark.slow
def test_the_achievable_residual_floor_is_measured_not_assumed(sizing_case):
    """Find the tightest tolerance the box can reach, and confirm the shipped case sits on it.

    The result is recorded in :doc:`/verification`: about 1e-9, independent of grid. The shipped
    cases request exactly that, which is why they converge and why asking for 1e-10 does not.
    """
    reached = {tol: _converges(sizing_case(), 11, tol) for tol in PROBED_TOLERANCES}

    # Loose tolerances must all be reachable, or something other than the floor is wrong.
    assert reached[1e-7] and reached[1e-8], f"the box cannot reach a loose tolerance: {reached}"
    # The shipped tolerance must be reachable, or the shipped cases are lucky rather than sound.
    assert reached[SHIPPED_TOLERANCE], "the shipped tolerance of 1e-9 is not reachable"
    # And there must be a floor below it, or the documented explanation of the grid study is wrong.
    assert not reached[1e-10], "the floor is tighter than 1e-10; the documented explanation is stale"


@pytest.mark.verification
@pytest.mark.slow
def test_the_answer_does_not_depend_on_how_tightly_it_was_converged(sizing_case):
    """Converging harder must not move the answer, only its residual.

    Between 1e-7 and 1e-9 the residual falls by two orders of magnitude. If the results moved
    with it, the reported digits would be solver artefacts rather than the solution.
    """
    loose = SizingAnalysis(Config.from_dict(_with_tolerance(sizing_case(), 1e-7))).run()
    tight = SizingAnalysis(Config.from_dict(_with_tolerance(sizing_case(), 1e-9))).run()

    for name in ("MTOW", "OEW", "total_fuel", "block_fuel", "takeoff_field_length", "V1"):
        assert float(loose[name]) == pytest.approx(float(tight[name]), rel=1e-7), name


def _with_tolerance(case: dict, tolerance: float) -> dict:
    """Return the case with its solver tolerances replaced."""
    case["black_box"]["num_nodes"] = 11
    case.setdefault("solver", {}).update({"maxiter": 80, "atol": tolerance, "rtol": tolerance})
    return case


@pytest.mark.verification
@pytest.mark.slow
def test_the_balanced_field_solve_is_a_solve_not_a_coincidence(sizing_case):
    """V1 must be driven by the residual, not left at its initial value.

    The box seeds ``takeoff|v1`` at 20 m/s and solves it implicitly so that the continue and
    abort distances match. A run that reported a plausible field length while V1 sat near its
    seed would have satisfied every other check in this repository.
    """
    results = SizingAnalysis(Config.from_dict(_with_tolerance(sizing_case(), 1e-9))).run()
    v1_seed_knots = 20.0 * 1.943844  # the box's declared initial value, in knots
    assert results["V1"] > 3.0 * v1_seed_knots, "V1 has not moved meaningfully from its seed"
    assert results["takeoff_field_length"] == pytest.approx(results["abort_distance"], rel=1e-6)
