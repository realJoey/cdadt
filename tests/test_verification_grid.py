"""Verification: discretization. Is the answer converged in the analysis grid?

The black box integrates fuel burn with Simpson's rule over ``num_nodes`` points per phase, and
solves each phase duration against an altitude or range target on that same grid. Every result
therefore carries a discretization error, and a number quoted without knowing that error is a
number quoted without knowing how many of its digits mean anything.

This module refines the grid and measures. It establishes three things:

1. The results converge monotonically under refinement.
2. The **observed order of convergence** is consistent with Simpson's rule, which is what says
   the integration is behaving as intended rather than accidentally agreeing.
3. The shipped grid of 21 nodes per phase is converged to a stated tolerance, so the digits
   quoted elsewhere in this repository are earned.

Run at a uniform solver tolerance of 1e-8 across every grid. That is deliberate: comparing
grids converged to different residuals would confound discretization error with solver error,
which is the most common way a grid study reaches a wrong conclusion.
"""

from __future__ import annotations

import math

import pytest

from cdadt import Config, SizingAnalysis

#: Grids to run. 21, 41 and 81 form a constant-refinement-ratio triple (r = 2), which is what
#: an observed order of convergence can be computed from.
GRIDS: tuple[int, ...] = (11, 21, 31, 41, 61, 81)

#: Triple used for the order estimate, coarse to fine, with a constant refinement ratio.
ORDER_TRIPLE: tuple[int, int, int] = (21, 41, 81)
REFINEMENT_RATIO = 2.0

#: A tolerance every grid can actually reach; see :mod:`tests.test_verification_solver` for the
#: measurement that fixes it. Uniform across grids so that only discretization varies.
SOLVER_TOLERANCE = 1e-8

#: Quantities the study tracks. These are the ones quoted in the documentation.
TRACKED = ("MTOW", "OEW", "block_fuel", "total_fuel", "takeoff_field_length", "V1")

#: The shipped grid must agree with the finest grid to better than this, relative.
SHIPPED_GRID_TOLERANCE = 1e-5


def _run(case: dict, num_nodes: int) -> dict[str, float]:
    """Converge the case on ``num_nodes`` points per phase and return the tracked results."""
    case["black_box"]["num_nodes"] = num_nodes
    case.setdefault("solver", {}).update({"maxiter": 60, "atol": SOLVER_TOLERANCE, "rtol": SOLVER_TOLERANCE})
    results = SizingAnalysis(Config.from_dict(case)).run()
    return {name: float(results[name]) for name in TRACKED}


@pytest.fixture(scope="module")
def refinement(sizing_case) -> dict[int, dict[str, float]]:
    """Converge the shipped case on every grid in :data:`GRIDS`."""
    return {num_nodes: _run(sizing_case(), num_nodes) for num_nodes in GRIDS}


@pytest.mark.verification
@pytest.mark.slow
def test_every_grid_converges(refinement):
    """Refinement must not itself be a source of failure.

    Recorded because it was not free: at the shipped tolerance of 1e-9 the 31- and 41-node
    grids stall at a residual of about 9e-9 and are reported as non-converged. The floor is a
    property of the box, not of the grid -- see :mod:`tests.test_verification_solver` -- and the
    study runs at 1e-8 for that reason.
    """
    assert set(refinement) == set(GRIDS)
    for num_nodes, results in refinement.items():
        assert results["MTOW"] > 0.0, f"grid {num_nodes} produced a nonphysical result"


@pytest.mark.verification
@pytest.mark.slow
def test_the_results_converge_under_refinement(refinement):
    """Each refinement must move the answer less than the one before it.

    Compared against the finest grid available. A sequence whose differences stop shrinking is
    not converging, whatever its individual values look like.
    """
    finest = refinement[GRIDS[-1]]
    not_converging = []
    for name in TRACKED:
        errors = [abs(refinement[n][name] - finest[name]) / abs(finest[name]) for n in GRIDS[:-1]]
        for coarse, fine, first, second in zip(GRIDS, GRIDS[1:], errors, errors[1:], strict=False):
            if second > first:
                not_converging.append(f"{name}: error grew from {first:.2e} at {coarse} to {second:.2e} at {fine}")

    assert not not_converging, "Refinement is not converging:\n  " + "\n  ".join(not_converging)


@pytest.mark.verification
@pytest.mark.slow
def test_the_observed_order_of_convergence_matches_simpsons_rule(refinement):
    """The measured order must be consistent with the integration scheme the box uses.

    With three grids at a constant refinement ratio :math:`r`, the observed order is

    .. math::

       p = \\frac{\\ln\\bigl((f_1 - f_2) / (f_2 - f_3)\\bigr)}{\\ln r}

    Simpson's rule is fourth order. Measuring roughly that is strong evidence the integration is
    doing what it claims; measuring first order would mean something in the chain had quietly
    become piecewise constant.

    The band is wide on purpose. The quantity being refined is not a bare quadrature: it is a
    quadrature inside a Newton-solved loop that also re-solves phase durations, so the measured
    order is an effective one.
    """
    coarse, middle, fine = ORDER_TRIPLE
    orders = {}
    for name in TRACKED:
        first = refinement[coarse][name] - refinement[middle][name]
        second = refinement[middle][name] - refinement[fine][name]
        if abs(second) < 1e-12 or abs(first) < 1e-12:
            continue  # already at the noise floor; the order is not measurable from it
        ratio = first / second
        if ratio <= 0:
            continue  # non-monotone in this quantity; excluded rather than reported as an order
        orders[name] = math.log(ratio) / math.log(REFINEMENT_RATIO)

    assert orders, "No tracked quantity gave a measurable order of convergence"
    for name, order in orders.items():
        assert 3.0 <= order <= 6.5, f"{name} converges at observed order {order:.2f}, not ~4 (Simpson)"


@pytest.mark.verification
@pytest.mark.slow
def test_the_shipped_grid_is_converged(refinement):
    """21 nodes per phase must agree with the finest grid to the documented tolerance.

    This is what licenses quoting MTOW and fuel to four decimal places elsewhere: not that the
    solver reports those digits, but that they do not move when the grid is refined fourfold.
    """
    finest = refinement[GRIDS[-1]]
    shipped = refinement[21]
    unconverged = []
    for name in TRACKED:
        error = abs(shipped[name] - finest[name]) / abs(finest[name])
        if error > SHIPPED_GRID_TOLERANCE:
            unconverged.append(f"{name}: {error:.2e} > {SHIPPED_GRID_TOLERANCE:.0e}")

    assert not unconverged, "The shipped 21-node grid is not converged to the documented tolerance:\n  " + "\n  ".join(
        unconverged
    )


@pytest.mark.verification
@pytest.mark.slow
def test_the_coarse_grid_used_for_optimization_is_fit_for_that_purpose(refinement):
    """11 nodes is what the optimization case runs on; its error must be known, not assumed.

    An optimization trades grid against iteration count, and 11 nodes is a defensible trade only
    if the discretization error is small against the design changes being resolved -- here a
    14% change in fuel. It is, by four orders of magnitude.
    """
    finest = refinement[GRIDS[-1]]
    error = abs(refinement[11]["total_fuel"] - finest["total_fuel"]) / abs(finest["total_fuel"])
    assert error < 1e-4, f"the optimization grid carries {error:.2e} discretization error in total fuel"
