"""Stability: sizing the empennage."""

from __future__ import annotations

from cdadt.core.discipline import Discipline, DisciplineScope

__all__ = ["Stability"]


class Stability(Discipline):
    """Empennage sizing.

    Aircraft-scoped. Tail areas are design quantities, and they feed both the drag buildup
    and the empty-weight buildup, so they must be computed once above the mission rather
    than per phase.
    """

    @property
    def name(self) -> str:
        """Return ``"stability"``."""
        return "stability"

    @property
    def scope(self) -> DisciplineScope:
        """Return :attr:`~cdadt.core.discipline.DisciplineScope.AIRCRAFT`."""
        return DisciplineScope.AIRCRAFT
