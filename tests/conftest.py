"""Shared fixtures.

Everything expensive is module- or session-scoped, because the two things that cost real time
-- converging the sizing case and converging OpenConcept's own example -- are each worth doing
once per session and comparing many times.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cdadt import Config, SizingAnalysis

#: Repository root, so tests address the shipped cases the way a user would.
ROOT = Path(__file__).resolve().parent.parent

#: Directory holding the shipped case files.
CASES = ROOT / "cases"


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
