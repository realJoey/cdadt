"""Pytest configuration for the cdadt test suite.

This module runs before any test module imports OpenMDAO, which is the only point at which
OpenMDAO's reporting system can be configured. Reports are disabled for the suite: they
write ``<script>_out/`` directories next to the working directory, which is noise during a
test run and would otherwise be mistaken for build output.

Nothing here changes model behavior. Report generation is a side output only; disabling it
does not alter any computed result.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

from pathlib import Path

import pytest

import cdadt
from cdadt.core.configuration import AircraftConfiguration

REPO_ROOT = Path(cdadt.__file__).resolve().parent.parent
B738_CONFIG_PATH = REPO_ROOT / "configs" / "b738.yml"


@pytest.fixture(scope="session")
def b738_config() -> AircraftConfiguration:
    """Return the shipped Boeing 737-800 configuration.

    The real file, not a fixture copy. A test that ran against a hand-built configuration
    would prove the code works on inputs written to suit it, and would not notice the
    shipped configuration going stale.
    """
    return AircraftConfiguration.from_yaml(B738_CONFIG_PATH)
