"""Landing performance: approach speed and field length at maximum landing weight."""

from __future__ import annotations

from cdadt.core.discipline import Discipline, DisciplineScope

__all__ = ["LandingPerformance"]


class LandingPerformance(Discipline):
    """Landing performance at maximum landing weight.

    Aircraft-scoped. The landing distance 14 CFR 25.125 asks for is demonstrated at maximum
    landing weight from a 50 ft screen height -- one condition, a property of the design,
    not a segment of the mission. OpenConcept's mission does not include it, so this
    discipline is how it enters the model at all.
    """

    @property
    def name(self) -> str:
        """Return ``"landing"``."""
        return "landing"

    @property
    def scope(self) -> DisciplineScope:
        """Return :attr:`~cdadt.core.discipline.DisciplineScope.AIRCRAFT`."""
        return DisciplineScope.AIRCRAFT
