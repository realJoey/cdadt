"""The geometry discipline: derived dimensions the rest of the model needs."""

from __future__ import annotations

from typing import ClassVar

from openconcept.geometry import CylinderSurfaceArea, WingMACTrapezoidal
from openconcept.utilities import AddSubtractComp

from cdadt.disciplines.base import AircraftDiscipline

__all__ = ["Geometry"]


class Geometry(AircraftDiscipline):
    """Dimensions derived from the primary shape parameters.

    Three derived quantities, each needed by another discipline:

    **Mean aerodynamic chord** of a trapezoidal wing, from reference area, aspect ratio and
    taper ratio. Used by the horizontal-tail volume-coefficient sizing and by the empty-weight
    buildup.

    **Tail lever arms**, estimated as a fraction of fuselage length -- the quarter-chord to
    quarter-chord distance from wing to tail. Required by both tail sizing and the empty
    weight buildup, and not otherwise derivable without a fuselage layout.

    **Wetted areas** of the fuselage and nacelles, treated as cylinders. Consumed by the
    parasite drag buildup and, for the fuselage, by the empty-weight buildup.

    Inputs
    ------
    ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|taper : float
        Wing planform (scalar).
    ac|geom|fuselage|length, ac|geom|fuselage|height : float
        Fuselage length and diameter (scalar, m).
    ac|geom|nacelle|length, ac|geom|nacelle|diameter : float
        Nacelle length and diameter (scalar, m).

    Outputs
    -------
    ac|geom|wing|MAC : float
        Mean aerodynamic chord (scalar, m).
    ac|geom|hstab|c4_to_wing_c4, ac|geom|vstab|c4_to_wing_c4 : float
        Tail lever arms (scalar, m).
    ac|geom|fuselage|S_wet, ac|geom|nacelle|S_wet : float
        Wetted areas (scalar, m**2).

    Notes
    -----
    :attr:`tail_arm_fraction` is a layout assumption, not a law. It is a class attribute
    rather than a buried literal so that a configuration with a real fuselage layout replaces
    it by subclassing, and so that its value appears in the model report.
    """

    discipline_name: ClassVar[str] = "geometry"

    #: Wing-to-tail quarter-chord distance as a fraction of fuselage length. 0.5 is the
    #: estimate OpenConcept's own B738 sizing example uses for a conventional tube-and-wing
    #: transport with aft-mounted empennage.
    tail_arm_fraction: ClassVar[float] = 0.5

    def setup(self) -> None:
        """Add the MAC, lever-arm and wetted-area calculations."""
        lever_arms = self.add_subsystem(
            "tail_lever_arms",
            AddSubtractComp(),
            promotes_inputs=[
                ("fuselage_length_hstab", "ac|geom|fuselage|length"),
                ("fuselage_length_vstab", "ac|geom|fuselage|length"),
            ],
            promotes_outputs=["ac|geom|hstab|c4_to_wing_c4", "ac|geom|vstab|c4_to_wing_c4"],
        )
        for tail in ("hstab", "vstab"):
            lever_arms.add_equation(
                output_name=f"ac|geom|{tail}|c4_to_wing_c4",
                input_names=[f"fuselage_length_{tail}"],
                units="m",
                scaling_factors=[self.tail_arm_fraction],
            )

        self.add_subsystem(
            "wing_mac",
            WingMACTrapezoidal(),
            promotes_inputs=[
                ("S_ref", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
            ],
            promotes_outputs=[("MAC", "ac|geom|wing|MAC")],
        )

        self.add_subsystem(
            "fuselage_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|fuselage|length"), ("D", "ac|geom|fuselage|height")],
            promotes_outputs=[("A", "ac|geom|fuselage|S_wet")],
        )
        self.add_subsystem(
            "nacelle_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|nacelle|length"), ("D", "ac|geom|nacelle|diameter")],
            promotes_outputs=[("A", "ac|geom|nacelle|S_wet")],
        )
