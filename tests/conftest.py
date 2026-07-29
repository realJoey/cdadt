"""Shared fixtures.

Everything expensive is module- or session-scoped, because the two things that cost real time
-- converging the sizing case and converging OpenConcept's own example -- are each worth doing
once per session and comparing many times.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from cdadt import Config, SizingAnalysis

#: Repository root, so tests address the shipped cases the way a user would.
ROOT = Path(__file__).resolve().parent.parent

#: Directory holding the shipped case files.
CASES = ROOT / "cases"


@pytest.fixture(scope="session", autouse=True)
def isolated_working_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Run the whole suite from a temporary directory, so it cannot write into the repository.

    Not hygiene for its own sake. A driver writes its own log -- pyOptSparse puts ``IPOPT.out``
    into the problem's output directory -- and OpenMDAO creates that directory on demand, named
    after the *problem*, in the working directory. ``reports=False`` does not prevent it: it
    suppresses the reports, not the directory a driver writes into. So every optimization test
    used to leave a ``__main__<n>_out`` folder in the repository root.

    The command line never hits this, because it always supplies a
    :class:`~cdadt.blackbox.RunDirectory` that names the problem and says where it goes. Tests
    construct analyses directly and mostly should not care, so the working directory is moved
    once for the session instead of every test having to remember.

    Everything the suite reads is addressed absolutely -- :data:`CASES`, the package, the
    OpenConcept clone -- so nothing depends on where it runs from.
    """
    directory = tmp_path_factory.mktemp("cdadt_session")
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(previous)


def case_dict(name: str) -> dict[str, Any]:
    """Return a shipped case file as a fresh, mutable dictionary.

    Verification studies vary one thing about a shipped case -- the grid, a tolerance, the
    continuation ladder -- and compare. Re-reading the YAML each time is what keeps one study
    from perturbing the next through a shared object.
    """
    with open(CASES / name, encoding="utf-8") as handle:
        return copy.deepcopy(yaml.safe_load(handle))


@pytest.fixture(scope="session")
def sizing_case() -> Callable[[], dict[str, Any]]:
    """Return a factory for fresh copies of the shipped sizing case, as a dictionary."""
    return lambda: case_dict("b738.yaml")


@pytest.fixture(scope="session")
def optimization_case() -> Callable[[], dict[str, Any]]:
    """Return a factory for fresh copies of the shipped optimization case, as a dictionary."""
    return lambda: case_dict("b738_optimization.yaml")


@pytest.fixture(scope="session")
def sizing_config() -> Config:
    """The shipped B738 sizing case, parsed."""
    return Config.from_yaml(CASES / "b738.yaml")


@pytest.fixture(scope="session")
def optimization_config() -> Config:
    """The shipped B738 certification-driven optimization case, parsed."""
    return Config.from_yaml(CASES / "b738_optimization.yaml")


@pytest.fixture(scope="session")
def converged_analysis(sizing_config: Config) -> SizingAnalysis:
    """A converged sizing analysis of the shipped B738 case, at the case's own grid."""
    analysis = SizingAnalysis(sizing_config)
    analysis.build()
    analysis.converge()
    return analysis


@pytest.fixture(scope="session")
def built_box(converged_analysis: SizingAnalysis):
    """The built and converged black box of the shipped case."""
    return converged_analysis.box


@pytest.fixture(scope="session")
def reference_problem(sizing_config: Config):
    """Run OpenConcept's own B738 sizing example at the shipped grid, and return its problem.

    Session-scoped because it is the most expensive fixture in the suite and two modules compare
    against it: :mod:`tests.test_validation`, which checks cdadt driving OpenConcept's own group,
    and :mod:`tests.test_adapter`, which checks cdadt driving its *own* group with its own
    aerodynamics. Both must reproduce the same reference, and running it twice would double the
    cost of the slowest thing here for no additional evidence.
    """
    from openconcept.examples.B738_sizing import run_738_sizing_analysis

    return run_738_sizing_analysis(num_nodes=sizing_config.black_box.num_nodes)
