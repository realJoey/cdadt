"""Landing performance provider.

OpenConcept models no landing, so this provider combines two of its components -- the flap
CLmax correlation and the stall-speed relation, the same ones used elsewhere in the model --
with cdadt's own approach-speed and field-length components.

The physics lives here rather than in :mod:`cdadt.certification` on purpose. A certification
requirement's job is to state a limit and read a value; if it also computed the value, the
regulation and the method would be entangled, and swapping the landing method would mean
editing the regulation. The boundary is enforced by
``cdadt/tests/mission/test_boundary.py``, which fails the suite if anything under
``cdadt/certification`` imports OpenConcept.

Reusing OpenConcept's ``FlapCLmax`` for the landing flap setting matters for a second
reason: takeoff CLmax comes from the same correlation, so the two cannot disagree about the
same wing.
"""

from __future__ import annotations

import openmdao.api as om
from openconcept.aerodynamics import FlapCLmax, StallSpeed

from cdadt.certification.components.landing import ApproachSpeed, LandingFieldLength
from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

__all__ = ["LandingPerformanceProvider"]


class LandingPerformanceProvider(Provider):
    """Approach speed and landing field length at maximum landing weight.

    Builds the chain: flap CLmax at the landing setting, stall speed at maximum landing
    weight, reference approach speed, and the factored landing field length.

    Notes
    -----
    **Required configuration**, none with defaults: ``ac|aero|landing_flap_deg``,
    ``certification|landing|approach_speed_factor``,
    ``certification|landing|airborne_distance``,
    ``certification|landing|mean_deceleration`` and
    ``certification|landing|dispatch_factor``.

    The stall speed is evaluated at maximum landing weight because that is the weight
    14 CFR 25.125 specifies for the landing distance demonstration. Evaluating it at
    takeoff weight would give a longer, more conservative distance -- and would not be what
    the regulation asks for.
    """

    REQUIRED_CONFIG = (
        "ac|aero|landing_flap_deg",
        "certification|landing|approach_speed_factor",
        "certification|landing|airborne_distance",
        "certification|landing|mean_deceleration",
        "certification|landing|dispatch_factor",
    )

    @property
    def name(self) -> str:
        """Return ``"landing_performance"``."""
        return "landing_performance"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "OpenConcept FlapCLmax and StallSpeed for the landing-configuration stall speed, with "
            "cdadt's ApproachSpeed (V_REF = k V_SR0, 14 CFR 25.125(a)(2)) and LandingFieldLength "
            "(airborne distance plus energy ground roll, factored per 14 CFR 121.195(b))."
        )

    def provides(self) -> VariableSet:
        """Return the landing performance quantities."""
        return VariableSet(
            [
                Variable("ac|aero|CLmax_landing", None, description="Maximum lift coefficient, landing flaps"),
                Variable("Vstall_land", "m/s", description="Reference stall speed, landing configuration"),
                Variable("V_ref", "m/s", description="Reference approach speed at the 50 ft height"),
                Variable("landing_distance", "m", description="Demonstrated landing distance from 50 ft"),
                Variable("landing_field_length", "m", description="Factored landing field length"),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the geometry, aerodynamics and weight this chain reads."""
        return VariableSet(
            [
                Variable("ac|aero|landing_flap_deg", "deg", description="Landing flap deflection"),
                Variable("ac|geom|wing|c4sweep", "rad", description="Wing quarter-chord sweep"),
                Variable("ac|geom|wing|toverc", None, description="Wing thickness-to-chord"),
                Variable("ac|aero|CLmax_cruise", None, description="Maximum lift coefficient, clean"),
                Variable("ac|geom|wing|S_ref", "m**2", description="Wing reference area"),
                Variable("ac|weights|MLW", "kg", description="Maximum landing mass"),
            ]
        )

    def validate_configuration(self) -> None:
        """Require the landing flap setting and every method constant."""
        self.config.require_all(list(self.REQUIRED_CONFIG))

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the landing performance chain to ``group``.

        ``num_nodes`` and ``flight_phase`` are unused: landing performance is evaluated at
        one condition, at maximum landing weight, and is a property of the design rather
        than of a mission phase.
        """
        constants = group.add_subsystem("constants", om.IndepVarComp(), promotes_outputs=["*"])
        for path, output in [
            ("certification|landing|approach_speed_factor", "approach_speed_factor"),
            ("certification|landing|airborne_distance", "airborne_distance"),
            ("certification|landing|mean_deceleration", "mean_deceleration"),
            ("certification|landing|dispatch_factor", "dispatch_factor"),
        ]:
            constants.add_output(output, val=self.config.value(path), units=self.config.units(path))

        group.add_subsystem(
            "landing_clmax",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|landing_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_landing")],
        )
        group.add_subsystem(
            "landing_stall_speed",
            StallSpeed(),
            promotes_inputs=[
                ("CLmax", "ac|aero|CLmax_landing"),
                ("weight", "ac|weights|MLW"),
                "ac|geom|wing|S_ref",
            ],
            promotes_outputs=[("Vstall_eas", "Vstall_land")],
        )
        group.add_subsystem(
            "approach_speed",
            ApproachSpeed(num_nodes=1),
            promotes_inputs=["Vstall_land", "approach_speed_factor"],
            promotes_outputs=["V_ref"],
        )
        group.add_subsystem(
            "field_length",
            LandingFieldLength(num_nodes=1),
            promotes_inputs=["V_ref", "airborne_distance", "mean_deceleration", "dispatch_factor"],
            promotes_outputs=["landing_distance", "landing_field_length"],
        )
