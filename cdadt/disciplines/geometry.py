"""Geometry: the shape of the airframe."""

from __future__ import annotations

from cdadt.core.discipline import Discipline, DisciplineScope

__all__ = ["Geometry"]


class Geometry(Discipline):
    """Airframe geometry: derived planform quantities and wetted areas.

    Aircraft-scoped. Geometry is a property of the design, identical at every point in the
    mission, so it is built once above the mission rather than inside each phase.

    Typical outputs are the wing mean aerodynamic chord, fuselage and nacelle wetted areas,
    and the tail moment arms that the stability discipline needs.
    """

    @property
    def name(self) -> str:
        """Return ``"geometry"``."""
        return "geometry"

    @property
    def scope(self) -> DisciplineScope:
        """Return :attr:`~cdadt.core.discipline.DisciplineScope.AIRCRAFT`."""
        return DisciplineScope.AIRCRAFT
