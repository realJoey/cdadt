"""Shared fixtures: the B738 case, and a small mission for tests that only need a model."""

from __future__ import annotations

from pathlib import Path

import pytest

from cdadt import AircraftDefinition, MissionProfile

CASES = Path(__file__).resolve().parent.parent / "cases"


@pytest.fixture
def aircraft() -> AircraftDefinition:
    """Return the B738 aircraft definition, freshly loaded so tests cannot affect each other."""
    return AircraftDefinition.from_yaml(CASES / "b738_aircraft.yaml")


@pytest.fixture
def profile() -> MissionProfile:
    """Return the B738 design mission and its continuation schedule."""
    return MissionProfile.from_yaml(CASES / "b738_mission.yaml")
