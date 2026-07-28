"""Aerodynamics: drag in flight, and maximum lift as a property of the airframe."""

from __future__ import annotations

from cdadt.core.discipline import Discipline, DisciplineScope

__all__ = ["Aerodynamics", "MaximumLift"]


class Aerodynamics(Discipline):
    """Aerodynamic forces at a flight condition.

    Phase-scoped: drag depends on dynamic pressure, lift coefficient, Reynolds number and
    the aircraft's configuration, all of which change through the mission. The provider is
    told which phase it is building into so it can apply the takeoff configuration -- flaps
    deployed, gear down -- where OpenConcept is running a ground roll or rotation.
    """

    @property
    def name(self) -> str:
        """Return ``"aerodynamics"``."""
        return "aerodynamics"


class MaximumLift(Discipline):
    """Maximum lift coefficients, clean and in high-lift configurations.

    Aircraft-scoped. ``CLmax`` depends on the airfoil, sweep and flap setting, not on the
    flight condition, and OpenConcept consumes it as a scalar design parameter
    (``ac|aero|CLmax_TO``) at the mission level to compute stall, rotation and V2 speeds.
    Computing it inside a phase would make it invisible at that level.
    """

    @property
    def name(self) -> str:
        """Return ``"maximum_lift"``."""
        return "maximum_lift"

    @property
    def scope(self) -> DisciplineScope:
        """Return :attr:`~cdadt.core.discipline.DisciplineScope.AIRCRAFT`."""
        return DisciplineScope.AIRCRAFT
