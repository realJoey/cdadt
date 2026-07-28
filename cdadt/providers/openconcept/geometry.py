"""Geometry providers wrapping OpenConcept's planform and wetted-area components."""

from __future__ import annotations

import openmdao.api as om
from openconcept.geometry import CylinderSurfaceArea, WingMACTrapezoidal
from openconcept.utilities import AddSubtractComp

from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

__all__ = ["TrapezoidalGeometryProvider"]


class TrapezoidalGeometryProvider(Provider):
    """Derived geometry for a trapezoidal wing with a cylindrical fuselage and nacelles.

    Computes the wing mean aerodynamic chord, the fuselage and nacelle wetted areas, and
    the tail moment arms that the empennage sizing and empty-weight buildups consume.

    Notes
    -----
    **Required configuration**, with no defaults:
    ``ac|geom|hstab|arm_fraction_of_fuselage`` and
    ``ac|geom|vstab|arm_fraction_of_fuselage``, the distance from the wing quarter chord to
    each tail's quarter chord as a fraction of fuselage length.

    OpenConcept's ``B738_sizing`` example hardcodes 0.5 for both. A fraction that is
    reasonable for one airframe is still a guess for another, so cdadt makes it an explicit
    input with a recorded source. The two are configured separately because a horizontal
    and a vertical tail need not share a moment arm, even when a first estimate gives them
    the same one.
    """

    @property
    def name(self) -> str:
        """Return ``"trapezoidal_geometry"``."""
        return "trapezoidal_geometry"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "OpenConcept openconcept.geometry.WingMACTrapezoidal (trapezoidal-planform MAC) and "
            "CylinderSurfaceArea (fuselage and nacelle wetted area as a cylinder). Tail moment arm "
            "estimated as a configured fraction of fuselage length."
        )

    def provides(self) -> VariableSet:
        """Return the derived geometry this provider creates."""
        return VariableSet(
            [
                Variable("ac|geom|wing|MAC", "m", description="Wing mean aerodynamic chord"),
                Variable("ac|geom|fuselage|S_wet", "m**2", description="Fuselage wetted area"),
                Variable("ac|geom|nacelle|S_wet", "m**2", description="Nacelle wetted area, one nacelle"),
                Variable(
                    "ac|geom|hstab|c4_to_wing_c4",
                    "m",
                    description="Horizontal stabilizer quarter chord to wing quarter chord",
                ),
                Variable(
                    "ac|geom|vstab|c4_to_wing_c4",
                    "m",
                    description="Vertical stabilizer quarter chord to wing quarter chord",
                ),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the primary geometry this provider reads."""
        return VariableSet(
            [
                Variable("ac|geom|wing|S_ref", "m**2", description="Wing reference area"),
                Variable("ac|geom|wing|AR", None, description="Wing aspect ratio"),
                Variable("ac|geom|wing|taper", None, description="Wing taper ratio"),
                Variable("ac|geom|fuselage|length", "m", description="Fuselage length"),
                Variable("ac|geom|fuselage|height", "m", description="Fuselage height"),
                Variable("ac|geom|nacelle|length", "m", description="Nacelle length"),
                Variable("ac|geom|nacelle|diameter", "m", description="Nacelle diameter"),
            ]
        )

    def validate_configuration(self) -> None:
        """Require both tail-arm fractions; there are no defaults for them."""
        self.config.require_all(
            [
                "ac|geom|hstab|arm_fraction_of_fuselage",
                "ac|geom|vstab|arm_fraction_of_fuselage",
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the geometry components to ``group``.

        ``num_nodes`` and ``flight_phase`` are unused: geometry does not vary with the
        flight condition, which is why this provider backs an aircraft-scoped discipline.
        """
        group.add_subsystem(
            "wing_mac",
            WingMACTrapezoidal(),
            promotes_inputs=[
                ("S_ref", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
            ],
            promotes_outputs=[("MAC", "ac|geom|wing|MAC")],
        )

        group.add_subsystem(
            "fuselage_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|fuselage|length"), ("D", "ac|geom|fuselage|height")],
            promotes_outputs=[("A", "ac|geom|fuselage|S_wet")],
        )
        group.add_subsystem(
            "nacelle_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|nacelle|length"), ("D", "ac|geom|nacelle|diameter")],
            promotes_outputs=[("A", "ac|geom|nacelle|S_wet")],
        )

        # Each tail's moment arm is a configured fraction of fuselage length.
        # AddSubtractComp with a scaling factor keeps the relationship analytic and
        # differentiable with respect to fuselage length, which matters because fuselage
        # length can be a design variable.
        for tail in ("hstab", "vstab"):
            group.add_subsystem(
                f"{tail}_moment_arm",
                AddSubtractComp(
                    output_name=f"ac|geom|{tail}|c4_to_wing_c4",
                    input_names=["fuselage_length"],
                    units="m",
                    scaling_factors=[self.config.scalar(f"ac|geom|{tail}|arm_fraction_of_fuselage")],
                ),
                promotes_inputs=[("fuselage_length", "ac|geom|fuselage|length")],
                promotes_outputs=[f"ac|geom|{tail}|c4_to_wing_c4"],
            )
