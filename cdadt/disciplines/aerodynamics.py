"""The aerodynamics discipline: zero-lift drag buildup and drag polar."""

from __future__ import annotations

from typing import ClassVar

from openconcept.aerodynamics import ParasiteDragCoefficient_JetTransport, PolarDrag

from cdadt.disciplines.base import PhaseDiscipline

__all__ = ["Aerodynamics"]


class Aerodynamics(PhaseDiscipline):
    """Drag at every node of one mission phase.

    Two OpenConcept components in series. A component-by-component parasite drag buildup
    (Roskam-style wetted-area and form-factor method, with high-lift and landing-gear
    increments when the aircraft is in takeoff configuration) produces :math:`C_{D_0}`; a
    parabolic polar adds the induced term:

    .. math::

       C_D = C_{D_0} + \\frac{C_L^2}{\\pi e \\mathrm{AR}},
       \\qquad D = q S_\\mathrm{ref} C_D

    :math:`C_{D_0}` is computed per node rather than once, because it depends on Reynolds
    number and therefore on speed, density and temperature.

    Inputs
    ------
    fltcond|Utrue, fltcond|rho, fltcond|T, fltcond|q, fltcond|CL : float
        Flight condition at each node, supplied by the OpenConcept phase (vector).
    ac|geom|*, ac|aero|polar|e, ac|propulsion|num_engines : float
        Airframe geometry and the span efficiency factor (scalar).
    ac|aero|takeoff_flap_deg, ac|geom|wing|c4sweep : float
        Read only in takeoff configuration, where the buildup adds flap and gear drag
        (scalar).

    Outputs
    -------
    drag : float
        Total drag force at each node (vector, N).

    Notes
    -----
    ``configuration`` is set from :attr:`~cdadt.disciplines.base.PhaseDiscipline.
    in_takeoff_configuration`, so which phases carry high-lift drag is a property of the
    phase model rather than a condition written into this class.
    """

    discipline_name: ClassVar[str] = "aerodynamics"

    #: Inputs the parasite drag buildup needs in every configuration.
    CLEAN_INPUTS: ClassVar[tuple[str, ...]] = (
        "fltcond|Utrue",
        "fltcond|rho",
        "fltcond|T",
        "ac|geom|fuselage|length",
        "ac|geom|fuselage|height",
        "ac|geom|fuselage|S_wet",
        "ac|geom|hstab|S_ref",
        "ac|geom|hstab|AR",
        "ac|geom|hstab|taper",
        "ac|geom|hstab|toverc",
        "ac|geom|vstab|S_ref",
        "ac|geom|vstab|AR",
        "ac|geom|vstab|taper",
        "ac|geom|vstab|toverc",
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|taper",
        "ac|geom|wing|toverc",
        "ac|geom|nacelle|length",
        "ac|geom|nacelle|S_wet",
        "ac|propulsion|num_engines",
    )

    #: Additional inputs the buildup needs when flaps and gear are deployed.
    TAKEOFF_INPUTS: ClassVar[tuple[str, ...]] = ("ac|aero|takeoff_flap_deg", "ac|geom|wing|c4sweep")

    def setup(self) -> None:
        """Add the parasite drag buildup and the drag polar, and join them."""
        num_nodes = self.options["num_nodes"]
        takeoff = self.in_takeoff_configuration

        promotes = list(self.CLEAN_INPUTS) + (list(self.TAKEOFF_INPUTS) if takeoff else [])
        self.add_subsystem(
            "zero_lift_drag",
            ParasiteDragCoefficient_JetTransport(
                num_nodes=num_nodes,
                configuration="takeoff" if takeoff else "clean",
            ),
            promotes_inputs=promotes,
        )

        self.add_subsystem(
            "drag_polar",
            PolarDrag(num_nodes=num_nodes, vec_CD0=True),
            promotes_inputs=[
                "fltcond|CL",
                "fltcond|q",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                ("e", "ac|aero|polar|e"),
            ],
            promotes_outputs=["drag"],
        )
        self.connect("zero_lift_drag.CD0", "drag_polar.CD0")
