"""Propulsion: thrust and fuel flow at a flight condition."""

from __future__ import annotations

from cdadt.core.discipline import Discipline

__all__ = ["Propulsion"]


class Propulsion(Discipline):
    """Installed thrust and fuel flow.

    Phase-scoped: thrust lapses with altitude and Mach number, and the number of operating
    propulsors changes between phases. OpenConcept sets ``propulsor_active`` to zero in the
    engine-failure phases that establish the balanced field length and the §25.121 climb
    gradient, so a provider that ignores it would size the aircraft against an
    all-engines-operating takeoff.
    """

    @property
    def name(self) -> str:
        """Return ``"propulsion"``."""
        return "propulsion"
