"""Thrust margin: the engine must be able to fly the mission it is sized for."""

from __future__ import annotations

from cdadt.certification.basis import Requirement, Sense
from cdadt.mission.blackbox import MissionBlackBox

__all__ = ["ThrottleMargin"]


class ThrottleMargin(Requirement):
    """Throttle must stay within its usable range in a given mission phase.

    Not a regulation, and it is labelled as such. It is the constraint that makes the
    others meaningful: OpenConcept's steady-flight phases solve for whatever throttle
    produces zero acceleration, and that solve is perfectly happy to return 1.4. The
    mission then converges on an aircraft flying at 140% of rated thrust, which is not a
    design -- and none of the certification requirements would notice, because each one
    reads a distance or a gradient that the over-thrusted aircraft achieves easily.

    Constraining throttle is therefore what turns "the mission converged" into "the engine
    can fly this mission".

    Parameters
    ----------
    config : AircraftConfiguration
        Configuration holding the throttle limit.
    phase : str
        Mission phase to constrain, as named in
        :data:`~cdadt.mission.contract.MISSION_PHASES`. One requirement per phase, so the
        traceability matrix reports which phase is short of thrust rather than only that
        something is.

    Notes
    -----
    **Required configuration**, with no default: ``certification|propulsion|throttle_max``.
    A limit slightly above 1.0 is conventional -- it lets a converging solver pass through
    the boundary without the constraint chattering -- but how far above is a modeling
    decision, not something to inherit.
    """

    LIMIT = "certification|propulsion|throttle_max"

    def __init__(self, config, phase: str) -> None:
        self._phase = phase
        super().__init__(config)

    @property
    def phase(self) -> str:
        """Return the mission phase this requirement constrains."""
        return self._phase

    @property
    def name(self) -> str:
        """Return a per-phase identifier."""
        return f"throttle_margin_{self._phase}"

    @property
    def regulation(self) -> str:
        """Return the label. This is a design constraint, not a regulation."""
        return "design"

    @property
    def title(self) -> str:
        """Return a one-line statement of the requirement."""
        return f"Throttle within limit in {self._phase}"

    @property
    def sense(self) -> str:
        """Return :attr:`~cdadt.certification.basis.Sense.UPPER`."""
        return Sense.UPPER

    @property
    def units(self) -> None:
        """Return ``None``: throttle is dimensionless."""
        return None

    def validate_configuration(self) -> None:
        """Require the throttle limit."""
        self.config.require_all([self.LIMIT])

    def limit(self) -> float:
        """Return the configured maximum throttle."""
        return self.config.scalar(self.LIMIT)

    @property
    def limit_source(self) -> str | None:
        """Return where the throttle limit came from."""
        return self.config.source(self.LIMIT)

    def constrained_path(self, blackbox: MissionBlackBox) -> str:
        """Return the path of this phase's throttle vector.

        The vector is constrained at every node, and
        :meth:`~cdadt.certification.basis.Requirement.evaluate` reports the worst one: a
        margin held on average but violated at top of climb is not a margin.
        """
        return blackbox.throttle_path(self._phase)
