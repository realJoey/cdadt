"""Shared fixtures for provider tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import cdadt
from cdadt.core.configuration import AircraftConfiguration

REPO_ROOT = Path(cdadt.__file__).resolve().parent.parent
B738_CONFIG_PATH = REPO_ROOT / "configs" / "b738.yml"


@pytest.fixture(scope="session")
def b738_config() -> AircraftConfiguration:
    """Return the shipped Boeing 737-800 configuration.

    The real file, not a fixture copy. A provider test that ran against a hand-built
    configuration would prove the provider works on inputs written to suit it, and would
    not notice the shipped configuration going stale.
    """
    return AircraftConfiguration.from_yaml(B738_CONFIG_PATH)
