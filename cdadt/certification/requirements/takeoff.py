"""Takeoff certification requirements: balanced field length and engine-out climb."""

from __future__ import annotations

from cdadt.certification.basis import Requirement, Sense
from cdadt.mission.blackbox import MissionBlackBox

__all__ = ["BalancedFieldLength", "EngineOutClimbGradient"]


class BalancedFieldLength(Requirement):
    """14 CFR 25.113: takeoff distance must fit the runway available.

    §25.113 defines the takeoff distance as the greater of the distance to reach 35 ft with
    the critical engine failed at V\\ :sub:`EF`, and 115% of the all-engines-operating
    distance. The balanced field is the condition where continuing and stopping cover the
    same distance, which is what V\\ :sub:`1` is chosen to achieve.

    OpenConcept already solves V\\ :sub:`1` implicitly so that the continue and abort
    distances match, so this requirement adds no physics -- it applies the limit. What it
    contributes is that the limit is *declared*: which runway the aircraft is being sized
    for is recorded in the configuration and reported in the traceability matrix, instead of
    being a number in a run script.

    Notes
    -----
    **Required configuration**, with no default:
    ``certification|takeoff|field_length_available``.
    """

    LIMIT = "certification|takeoff|field_length_available"

    @property
    def name(self) -> str:
        """Return ``"far25_113_balanced_field_length"``."""
        return "far25_113_balanced_field_length"

    @property
    def regulation(self) -> str:
        """Return the regulation citation."""
        return "14 CFR 25.113"

    @property
    def title(self) -> str:
        """Return a one-line statement of the requirement."""
        return "Takeoff distance within field available"

    @property
    def sense(self) -> str:
        """Return :attr:`~cdadt.certification.basis.Sense.UPPER`: distance must not exceed."""
        return Sense.UPPER

    @property
    def units(self) -> str:
        """Return ``"ft"``."""
        return "ft"

    def validate_configuration(self) -> None:
        """Require the available field length."""
        self.config.require_all([self.LIMIT])

    def limit(self) -> float:
        """Return the configured field length available, in ft."""
        return self.config.scalar(self.LIMIT, units=self.units)

    @property
    def limit_source(self) -> str | None:
        """Return where the field length came from."""
        return self.config.source(self.LIMIT)

    def constrained_path(self, blackbox: MissionBlackBox) -> str:
        """Return the path of the balanced field length."""
        return blackbox.path("takeoff_field_length")


class EngineOutClimbGradient(Requirement):
    """14 CFR 25.121(b): second-segment climb gradient with the critical engine inoperative.

    §25.121(b) requires a steady gradient of climb at V\\ :sub:`2`, with the critical engine
    inoperative, the landing gear retracted and takeoff flaps set, of at least 2.4% for
    two-engine aeroplanes, 2.7% for three, and 3.0% for four. Which of those applies is a
    property of the aircraft, so the required gradient is configured rather than inferred.

    OpenConcept's mission already includes the engine-out climb-angle condition, evaluated
    at V\\ :sub:`2` with ``propulsor_active`` set to zero. This requirement constrains its
    output.

    Notes
    -----
    The regulation states a *gradient*: rise over run, a tangent. OpenConcept reports the
    climb *angle* in radians. These differ by ``tan``, and for a 2.4% gradient the
    difference is under 0.1% of the value -- but it is a real difference, and this
    requirement converts rather than conflating the two.

    **Required configuration**, with no default:
    ``certification|climb|oei_second_segment_gradient``.
    """

    LIMIT = "certification|climb|oei_second_segment_gradient"

    @property
    def name(self) -> str:
        """Return ``"far25_121_oei_climb_gradient"``."""
        return "far25_121_oei_climb_gradient"

    @property
    def regulation(self) -> str:
        """Return the regulation citation."""
        return "14 CFR 25.121(b)"

    @property
    def title(self) -> str:
        """Return a one-line statement of the requirement."""
        return "OEI second-segment climb gradient"

    @property
    def sense(self) -> str:
        """Return :attr:`~cdadt.certification.basis.Sense.LOWER`: gradient must not fall below."""
        return Sense.LOWER

    @property
    def units(self) -> str:
        """Return ``"rad"``.

        The constrained quantity is OpenConcept's climb angle, which is an angle. The
        configured limit is a gradient and is converted to the equivalent angle by
        :meth:`limit`.
        """
        return "rad"

    def validate_configuration(self) -> None:
        """Require the climb gradient."""
        self.config.require_all([self.LIMIT])

    def limit(self) -> float:
        """Return the required climb angle in radians, converted from the configured gradient.

        Returns
        -------
        float
            ``arctan(gradient)``. Constraining the angle to ``arctan(g)`` is exactly
            equivalent to constraining the gradient to ``g``, because ``tan`` is monotonic
            over the range of interest.
        """
        import numpy as np

        return float(np.arctan(self.config.scalar(self.LIMIT)))

    @property
    def limit_source(self) -> str | None:
        """Return where the required gradient came from."""
        return self.config.source(self.LIMIT)

    def constrained_path(self, blackbox: MissionBlackBox) -> str:
        """Return the path of the engine-out climb angle."""
        return blackbox.path("engine_out_climb_gradient")
