"""Landing certification requirements: approach speed and landing field length.

OpenConcept models no landing, so unlike the takeoff requirements these contribute their
own components. The chain built here is:

``FlapCLmax`` at the landing flap setting -> ``StallSpeed`` at maximum landing weight ->
:class:`~cdadt.certification.components.landing.ApproachSpeed` ->
:class:`~cdadt.certification.components.landing.LandingFieldLength`.

The two OpenConcept components in that chain are the ones already used elsewhere in the
model, so landing CLmax is computed by the same method as takeoff CLmax rather than by a
second one that could disagree with it.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from cdadt.certification.basis import Requirement, Sense
from cdadt.mission.blackbox import MissionBlackBox

__all__ = ["LANDING_GROUP", "ApproachSpeedLimit", "LandingFieldLengthLimit"]

#: Subsystem name the landing chain is built under, at the top of the model.
LANDING_GROUP = "landing_performance"


class _LandingRequirement(Requirement):
    """Shared behavior for requirements evaluated from the landing performance chain.

    The physics lives in
    :class:`~cdadt.providers.openconcept.landing.LandingPerformanceProvider`, not here. A
    requirement states a limit and reads a value; if it also computed the value, the
    regulation and the method would be entangled and swapping the landing method would mean
    editing the regulation. The provider is imported inside :meth:`build` rather than at
    module scope so that ``cdadt.certification`` carries no import-time dependency on any
    particular physics implementation.
    """

    #: Configuration the landing method needs, mirrored from the provider so a requirement
    #: can validate its configuration without importing the provider.
    METHOD_CONFIG = (
        "ac|aero|landing_flap_deg",
        "certification|landing|approach_speed_factor",
        "certification|landing|airborne_distance",
        "certification|landing|mean_deceleration",
        "certification|landing|dispatch_factor",
    )

    def build(self, model: om.Group, blackbox: MissionBlackBox) -> None:
        """Add the landing chain, unless another landing requirement already did.

        Several requirements read the same chain, and building it twice would give them two
        independently evaluated approach speeds. The traceability matrix would then report a
        field length and an approach speed that do not describe the same landing.
        """
        if model._get_subsystem(LANDING_GROUP) is not None:
            return

        from cdadt.core.discipline import DisciplineGroup
        from cdadt.disciplines.landing import LandingPerformance
        from cdadt.providers.openconcept.landing import LandingPerformanceProvider

        discipline = LandingPerformance(LandingPerformanceProvider(self.config), self.config)
        model.add_subsystem(
            LANDING_GROUP,
            DisciplineGroup(
                disciplines=[discipline],
                config=self.config,
                num_nodes=1,
                flight_phase="landing",
            ),
            promotes_inputs=["ac|*"],
        )


class LandingFieldLengthLimit(_LandingRequirement):
    """14 CFR 25.125 with 121.195(b): landing field length within the runway available.

    §25.125 defines the landing distance as the horizontal distance from 50 ft above the
    landing surface to a full stop, at maximum landing weight. §121.195(b) then requires
    dispatch only to an airport where that distance fits within 60% of the effective runway
    length. This requirement constrains the factored distance.

    Notes
    -----
    **Required configuration**, with no defaults:
    ``certification|landing|field_length_available`` for the limit, plus every entry in
    ``METHOD_CONFIG`` for the landing method itself.
    """

    LIMIT = "certification|landing|field_length_available"

    @property
    def name(self) -> str:
        """Return ``"far25_125_landing_field_length"``."""
        return "far25_125_landing_field_length"

    @property
    def regulation(self) -> str:
        """Return the regulation citation."""
        return "14 CFR 25.125"

    @property
    def title(self) -> str:
        """Return a one-line statement of the requirement."""
        return "Landing field length within available"

    @property
    def sense(self) -> str:
        """Return :attr:`~cdadt.certification.basis.Sense.UPPER`."""
        return Sense.UPPER

    @property
    def units(self) -> str:
        """Return ``"ft"``."""
        return "ft"

    def validate_configuration(self) -> None:
        """Require the limit and every method constant the landing chain needs."""
        self.config.require_all([self.LIMIT, *self.METHOD_CONFIG])

    def limit(self) -> float:
        """Return the configured landing field length available, in ft."""
        return self.config.scalar(self.LIMIT, units=self.units)

    @property
    def limit_source(self) -> str | None:
        """Return where the landing field length came from."""
        return self.config.source(self.LIMIT)

    def constrained_path(self, blackbox: MissionBlackBox) -> str:
        """Return the path of the factored landing field length."""
        return f"{LANDING_GROUP}.landing_field_length"


class ApproachSpeedLimit(_LandingRequirement):
    """Approach speed within the limit for the intended approach category.

    ICAO and FAA approach categories are defined by V\\ :sub:`REF` at maximum landing
    weight: category C spans 121 to 140 kn, category D 141 to 165 kn. The category
    determines which approach procedures and which airports an aircraft can use, so it is a
    design requirement even though no single regulation states it as a limit.

    The limit is configured for that reason -- there is no universal number, only the one
    implied by the category an operator needs.

    Notes
    -----
    **Required configuration**, with no default:
    ``certification|landing|approach_speed_max``, plus the landing method constants.
    """

    LIMIT = "certification|landing|approach_speed_max"

    @property
    def name(self) -> str:
        """Return ``"approach_speed_limit"``."""
        return "approach_speed_limit"

    @property
    def regulation(self) -> str:
        """Return the citation. The category definition, not a limit-setting regulation."""
        return "14 CFR 97 / ICAO cat"

    @property
    def title(self) -> str:
        """Return a one-line statement of the requirement."""
        return "Approach speed within category"

    @property
    def sense(self) -> str:
        """Return :attr:`~cdadt.certification.basis.Sense.UPPER`."""
        return Sense.UPPER

    @property
    def units(self) -> str:
        """Return ``"kn"``."""
        return "kn"

    def validate_configuration(self) -> None:
        """Require the limit, the method constants, and check the regulatory floor.

        §25.125(a)(2) sets ``V_REF >= 1.23 V_SR0``. A configuration that asked for less
        would describe an aircraft that cannot be certified, and the resulting landing
        distances would be shorter than any real aircraft achieves -- so this is rejected
        rather than flown.
        """
        self.config.require_all([self.LIMIT, *self.METHOD_CONFIG])

        factor = self.config.scalar("certification|landing|approach_speed_factor")
        regulatory_floor = 1.23
        if factor < regulatory_floor:
            raise ValueError(
                f"certification|landing|approach_speed_factor is {factor}, below the {regulatory_floor} "
                f"floor set by 14 CFR 25.125(a)(2) (V_REF >= 1.23 V_SR0). A landing distance computed "
                f"from a lower approach speed is shorter than any certifiable aircraft achieves."
            )

    def limit(self) -> float:
        """Return the configured maximum approach speed, in kn."""
        return self.config.scalar(self.LIMIT, units=self.units)

    @property
    def limit_source(self) -> str | None:
        """Return where the approach speed limit came from."""
        return self.config.source(self.LIMIT)

    def constrained_path(self, blackbox: MissionBlackBox) -> str:
        """Return the path of the reference approach speed."""
        return f"{LANDING_GROUP}.V_ref"


def landing_stall_speed_from(problem: om.Problem) -> float:
    """Return the landing-configuration stall speed from a run problem, in kn.

    Convenience for reports: the stall speed is what both landing requirements ultimately
    rest on, so it is worth reporting alongside them.
    """
    return float(np.asarray(problem.get_val(f"{LANDING_GROUP}.Vstall_land", units="kn")).reshape(-1)[0])
