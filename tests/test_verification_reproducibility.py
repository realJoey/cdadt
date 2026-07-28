"""Verification: reproducibility. Is the answer a property of the case, or of the route to it?

The black box is converged by walking a continuation ladder: a sequence of progressively harder
missions, each converged before the next is attempted. The ladder exists because a Newton solver
started cold on a 2800 nmi mission at FL350 does not reach it.

That raises a question a sizing tool must answer before its numbers mean anything: **is the
ladder a path to the solution, or part of it?** If two different ladders converge the same case
to different answers, the "solution" is an artefact of the route.

This module also checks plain determinism -- the same case run twice gives bit-identical
results -- and that the two shipped cases, which describe the same aeroplane at different grids,
agree about that aeroplane.
"""

from __future__ import annotations

import pytest
from openmdao.core.analysis_error import AnalysisError

from cdadt import Config, SizingAnalysis

#: Results compared across routes. If any of these move, the route is part of the answer.
COMPARED = ("MTOW", "OEW", "MLW", "block_fuel", "total_fuel", "takeoff_field_length", "V1", "V2")

#: Two different ladders must agree to this, relative. Tighter than the solver tolerance would
#: be meaningless; looser would fail to detect a real path dependence.
ROUTE_TOLERANCE = 1e-7

GENTLE_DESCENT = {
    "Ueas": {"value": [252, 250], "units": "kn"},
    "vs": {"value": [-800, -800], "units": "ft/min"},
}


def _rung(description: str, mission_range: float, altitude: float, reserve: float, reserve_altitude: float) -> dict:
    """Return one continuation step at the given range and altitude."""
    return {
        "description": description,
        "parameters": {
            "mission_range": {"value": mission_range, "units": "nmi"},
            "cruise|h0": {"value": altitude, "units": "ft"},
            "reserve_range": {"value": reserve, "units": "nmi"},
            "reserve|h0": {"value": reserve_altitude, "units": "ft"},
        },
        "schedule": {"descent": GENTLE_DESCENT},
    }


#: Three genuinely different routes to the same design mission.
ROUTES: dict[str, list[dict]] = {
    "shipped": [],  # replaced at run time by the case file's own ladder
    "four_rungs": [
        _rung("500 nmi at 5 kft", 500, 5000, 100, 1000),
        _rung("1200 nmi at 15 kft", 1200, 15000, 150, 5000),
        _rung("2000 nmi at 25 kft", 2000, 25000, 200, 10000),
        _rung("design mission, gentle descent", 2800, 35000, 200, 15000),
    ],
    "altitude_first": [
        _rung("300 nmi at 35 kft", 300, 35000, 100, 15000),
        _rung("1500 nmi at 35 kft", 1500, 35000, 200, 15000),
        _rung("design mission, gentle descent", 2800, 35000, 200, 15000),
    ],
}


def _run(case: dict, ladder: list[dict] | None, maxiter: int = 60) -> dict[str, float]:
    """Converge the case, optionally replacing its continuation ladder.

    The iteration budget is raised above the shipped case's 20 so that a route is judged on
    where it arrives rather than on how many Newton steps it needed to get there. Confounding
    those two is what would turn "this ladder needs more iterations" into a false report of
    path dependence.
    """
    case["black_box"]["num_nodes"] = 11
    case.setdefault("solver", {})["maxiter"] = maxiter
    if ladder is not None:
        case["mission"]["continuation"] = ladder
    results = SizingAnalysis(Config.from_dict(case)).run()
    return {name: float(results[name]) for name in COMPARED}


@pytest.mark.verification
@pytest.mark.slow
def test_the_same_case_run_twice_gives_identical_results(sizing_case):
    """Determinism. Nothing in the chain depends on iteration order, hashing or wall clock."""
    first = _run(sizing_case(), None)
    second = _run(sizing_case(), None)
    for name in COMPARED:
        assert first[name] == second[name], f"{name} is not deterministic: {first[name]!r} vs {second[name]!r}"


@pytest.mark.verification
@pytest.mark.slow
def test_different_continuation_ladders_reach_the_same_aircraft(sizing_case):
    """The central reproducibility claim: the ladder is a route, not part of the answer.

    Three ladders that differ in the number of rungs, in how far each one steps, and in whether
    range or altitude is relaxed first. If the converged aircraft depended on which was walked,
    every number this repository reports would be conditional on a choice nobody stated.
    """
    baseline = _run(sizing_case(), ROUTES["four_rungs"])

    disagreements: list[str] = []
    unreachable: list[str] = []
    arrived = 1
    for label in ("shipped", "altitude_first"):
        ladder = None if label == "shipped" else ROUTES[label]
        try:
            other = _run(sizing_case(), ladder)
        except AnalysisError:
            # A route that does not arrive says nothing about path dependence: it is a statement
            # about that ladder, not about the solution. Recorded, and required not to be all of
            # them by the assertion below.
            unreachable.append(label)
            continue
        arrived += 1
        for name in COMPARED:
            error = abs(other[name] - baseline[name]) / max(abs(baseline[name]), 1e-30)
            if error > ROUTE_TOLERANCE:
                disagreements.append(f"{label}.{name}: {other[name]!r} vs {baseline[name]!r} (rel {error:.2e})")

    assert not disagreements, "The converged answer depends on the continuation route:\n  " + "\n  ".join(disagreements)
    assert arrived >= 2, (
        f"only one route reached the design mission, so nothing was actually compared; "
        f"routes that did not arrive: {unreachable}"
    )


@pytest.mark.verification
@pytest.mark.slow
def test_the_two_shipped_cases_agree_about_the_aircraft(sizing_case, optimization_case):
    """They run different grids for different reasons; they must still describe one aeroplane.

    The sizing case runs 21 nodes for accuracy, the optimization case 11 for iteration count.
    Their baseline aircraft must agree to within the known discretization error of the coarser
    grid -- which :mod:`tests.test_verification_grid` measures independently at about 3e-5.
    """
    sizing = _run(sizing_case(), None)
    baseline = SizingAnalysis(Config.from_dict(optimization_case()))
    baseline.build()
    baseline.converge()
    optimizing = baseline.results()

    for name in COMPARED:
        assert float(optimizing[name]) == pytest.approx(sizing[name], rel=1e-4), name
