"""Aerodynamics providers wrapping OpenConcept's jet-transport drag and CLmax buildups."""

from __future__ import annotations

import openmdao.api as om
from openconcept.aerodynamics import CleanCLmax, FlapCLmax, ParasiteDragCoefficient_JetTransport, PolarDrag

from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet
from cdadt.mission.contract import DRAG, PHASE_SPECS_BY_NAME

__all__ = ["JetTransportDragProvider", "JetTransportMaximumLiftProvider"]

#: Geometry the parasite-drag buildup reads in every configuration.
_DRAG_GEOMETRY = [
    Variable("ac|geom|fuselage|length", "m", description="Fuselage length"),
    Variable("ac|geom|fuselage|height", "m", description="Fuselage height"),
    Variable("ac|geom|fuselage|S_wet", "m**2", description="Fuselage wetted area"),
    Variable("ac|geom|hstab|S_ref", "m**2", description="Horizontal stabilizer area"),
    Variable("ac|geom|hstab|AR", None, description="Horizontal stabilizer aspect ratio"),
    Variable("ac|geom|hstab|taper", None, description="Horizontal stabilizer taper ratio"),
    Variable("ac|geom|hstab|toverc", None, description="Horizontal stabilizer thickness-to-chord"),
    Variable("ac|geom|vstab|S_ref", "m**2", description="Vertical stabilizer area"),
    Variable("ac|geom|vstab|AR", None, description="Vertical stabilizer aspect ratio"),
    Variable("ac|geom|vstab|taper", None, description="Vertical stabilizer taper ratio"),
    Variable("ac|geom|vstab|toverc", None, description="Vertical stabilizer thickness-to-chord"),
    Variable("ac|geom|wing|S_ref", "m**2", description="Wing reference area"),
    Variable("ac|geom|wing|AR", None, description="Wing aspect ratio"),
    Variable("ac|geom|wing|taper", None, description="Wing taper ratio"),
    Variable("ac|geom|wing|toverc", None, description="Wing thickness-to-chord"),
    Variable("ac|geom|nacelle|length", "m", description="Nacelle length"),
    Variable("ac|geom|nacelle|S_wet", "m**2", description="Nacelle wetted area"),
    Variable("ac|propulsion|num_engines", None, description="Number of engines"),
]

#: Extra inputs the buildup reads only when the aircraft is in takeoff configuration.
_TAKEOFF_CONFIGURATION = [
    Variable("ac|aero|takeoff_flap_deg", "deg", description="Takeoff flap deflection"),
    Variable("ac|geom|wing|c4sweep", "rad", description="Wing quarter-chord sweep"),
]


class JetTransportDragProvider(Provider):
    """Drag from a parasite-drag buildup plus a parabolic induced-drag polar.

    The buildup runs in **takeoff configuration** -- flaps deployed, with the associated
    profile and induced drag increments -- for the four balanced-field phases, and clean
    everywhere else. That branch is what makes the balanced field length and the §25.121
    climb gradient reflect the configuration the regulation is written about; running clean
    drag through the takeoff phases would understate both.

    Notes
    -----
    **Required configuration.** ``ac|aero|polar|e``, the Oswald span efficiency. Required,
    with no default.
    """

    @property
    def name(self) -> str:
        """Return ``"jet_transport_drag"``."""
        return "jet_transport_drag"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "OpenConcept ParasiteDragCoefficient_JetTransport, a component buildup after Roskam, "
            "Raymer and the OpenVSP parasite drag method, combined with PolarDrag "
            "(CD = CD0 + CL^2 / (pi e AR))."
        )

    def provides(self) -> VariableSet:
        """Return the drag force and the zero-lift drag coefficient behind it."""
        return VariableSet(
            [
                DRAG,
                Variable("CD0", None, vectorized=True, description="Zero-lift drag coefficient"),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the flight conditions and geometry this buildup reads.

        The takeoff-configuration inputs are declared unconditionally even though the
        buildup reads them only in the four takeoff phases. Declaring what the provider can
        read is the honest statement; both are ``ac|`` design parameters promoted from the
        top of the mission, so they are available in every phase whether or not that phase's
        buildup consumes them.
        """
        flight_conditions = [
            Variable("fltcond|Utrue", "m/s", vectorized=True, description="True airspeed"),
            Variable("fltcond|rho", "kg/m**3", vectorized=True, description="Air density"),
            Variable("fltcond|T", "degK", vectorized=True, description="Air temperature"),
            Variable("fltcond|q", "N/m**2", vectorized=True, description="Dynamic pressure"),
            Variable("fltcond|CL", None, vectorized=True, description="Lift coefficient"),
        ]
        # Declaring the Oswald efficiency is not optional bookkeeping. It is a promoted
        # input of PolarDrag with its own component-level default, so a provider that did
        # not declare it would silently compute induced drag from that default instead of
        # from the configured value, and every drag number would be quietly wrong.
        oswald_efficiency = Variable("ac|aero|polar|e", None, description="Oswald span efficiency")
        return VariableSet([*flight_conditions, *_DRAG_GEOMETRY, *_TAKEOFF_CONFIGURATION, oswald_efficiency])

    def validate_configuration(self) -> None:
        """Require the Oswald efficiency; there is no default for it."""
        self.config.require_all(["ac|aero|polar|e"])

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the drag buildup and polar to ``group``, in the phase's configuration."""
        spec = PHASE_SPECS_BY_NAME[flight_phase]
        configuration = "takeoff" if spec.in_takeoff else "clean"

        promoted = [variable.name for variable in _DRAG_GEOMETRY]
        promoted += ["fltcond|Utrue", "fltcond|rho", "fltcond|T"]
        if spec.in_takeoff:
            promoted += [variable.name for variable in _TAKEOFF_CONFIGURATION]

        group.add_subsystem(
            "zero_lift_drag",
            ParasiteDragCoefficient_JetTransport(num_nodes=num_nodes, configuration=configuration),
            promotes_inputs=promoted,
            promotes_outputs=["CD0"],
        )

        group.add_subsystem(
            "drag_polar",
            PolarDrag(num_nodes=num_nodes, vec_CD0=True),
            promotes_inputs=[
                "fltcond|CL",
                "fltcond|q",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "CD0",
                ("e", "ac|aero|polar|e"),
            ],
            promotes_outputs=["drag"],
        )


class JetTransportMaximumLiftProvider(Provider):
    """Maximum lift coefficients, clean and with takeoff flaps.

    Aircraft-scoped. OpenConcept consumes ``ac|aero|CLmax_TO`` at the mission level to
    compute stall speed, rotation speed and V2, so it must exist as a scalar design
    parameter above the phases rather than inside one.
    """

    @property
    def name(self) -> str:
        """Return ``"jet_transport_maximum_lift"``."""
        return "jet_transport_maximum_lift"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "OpenConcept CleanCLmax (airfoil Clmax swept back to wing CLmax) and FlapCLmax "
            "(flap increment as a function of deflection, sweep and thickness-to-chord)."
        )

    def provides(self) -> VariableSet:
        """Return the clean and takeoff maximum lift coefficients."""
        return VariableSet(
            [
                Variable("ac|aero|CLmax_cruise", None, description="Maximum lift coefficient, clean"),
                Variable("ac|aero|CLmax_TO", None, description="Maximum lift coefficient, takeoff flaps"),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the airfoil and planform properties this method reads."""
        return VariableSet(
            [
                Variable("ac|aero|airfoil_Cl_max", None, description="Airfoil section maximum lift coefficient"),
                Variable("ac|geom|wing|c4sweep", "rad", description="Wing quarter-chord sweep"),
                Variable("ac|geom|wing|toverc", None, description="Wing thickness-to-chord"),
                Variable("ac|aero|takeoff_flap_deg", "deg", description="Takeoff flap deflection"),
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the two CLmax components to ``group``.

        ``num_nodes`` and ``flight_phase`` are unused: maximum lift is a property of the
        airframe and its high-lift system, not of a flight condition.
        """
        group.add_subsystem(
            "clean_clmax",
            CleanCLmax(),
            promotes_inputs=["ac|aero|airfoil_Cl_max", "ac|geom|wing|c4sweep"],
            promotes_outputs=[("CL_max_clean", "ac|aero|CLmax_cruise")],
        )
        group.add_subsystem(
            "takeoff_clmax",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|takeoff_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_TO")],
        )
