"""Weights: the empty-weight buildup, and the mass carried through the mission."""

from __future__ import annotations

from cdadt.core.discipline import Discipline, DisciplineScope

__all__ = ["EmptyWeight", "MassBookkeeping"]


class EmptyWeight(Discipline):
    """Operating empty weight from a component buildup.

    Aircraft-scoped. The buildup depends on geometry, engine rating and design weights, not
    on the flight condition. Its output closes the sizing loop:
    ``MTOW = OEW + payload + fuel``, where the fuel comes back from the mission.
    """

    @property
    def name(self) -> str:
        """Return ``"empty_weight"``."""
        return "empty_weight"

    @property
    def scope(self) -> DisciplineScope:
        """Return :attr:`~cdadt.core.discipline.DisciplineScope.AIRCRAFT`."""
        return DisciplineScope.AIRCRAFT


class MassBookkeeping(Discipline):
    """Instantaneous mass: takeoff mass less the fuel burned so far.

    Phase-scoped, and the one discipline whose state genuinely crosses phase boundaries.
    OpenConcept's ``PhaseGroup`` walks the aircraft model looking for ``Integrator``
    instances, connects the phase duration to them, and links their endpoint states across
    phases -- which is how fuel burned in climb carries into cruise. A provider that
    integrated fuel by some other means would restart from zero in every phase, and the
    total fuel that closes the sizing loop would be the loiter fuel alone.
    """

    @property
    def name(self) -> str:
        """Return ``"mass"``."""
        return "mass"
