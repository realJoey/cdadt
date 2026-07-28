"""The high-lift discipline: maximum lift coefficients, clean and with takeoff flaps."""

from __future__ import annotations

from typing import ClassVar

from openconcept.aerodynamics import CleanCLmax, FlapCLmax

from cdadt.disciplines.base import AircraftDiscipline

__all__ = ["HighLift"]


class HighLift(AircraftDiscipline):
    """Maximum lift coefficients of the wing, clean and in takeoff configuration.

    Two correlations in series. The clean maximum is the airfoil's section maximum reduced for
    quarter-chord sweep; the flapped maximum adds a flap increment that depends on the flap
    deflection, the sweep and the thickness ratio.

    These are the coefficients the certification-relevant speeds are built on. OpenConcept's
    takeoff phases use :math:`C_{L_{max,TO}}` to set the stall speed, and from it the rotation
    speed and the V\\ :sub:`2` at which the engine-out climb gradient is evaluated -- so this
    discipline is upstream of two of the requirements in
    :mod:`cdadt.certification`.

    Inputs
    ------
    ac|aero|airfoil_Cl_max : float
        Section maximum lift coefficient (scalar).
    ac|geom|wing|c4sweep, ac|geom|wing|toverc : float
        Quarter-chord sweep and thickness-to-chord ratio (scalar).
    ac|aero|takeoff_flap_deg : float
        Takeoff flap deflection (scalar, deg).

    Outputs
    -------
    ac|aero|CLmax_cruise : float
        Clean maximum lift coefficient (scalar).
    ac|aero|CLmax_TO : float
        Maximum lift coefficient with takeoff flaps (scalar).
    """

    discipline_name: ClassVar[str] = "high_lift"

    def setup(self) -> None:
        """Add the clean and flapped maximum-lift correlations."""
        self.add_subsystem(
            "clean",
            CleanCLmax(),
            promotes_inputs=["ac|aero|airfoil_Cl_max", "ac|geom|wing|c4sweep"],
            promotes_outputs=[("CL_max_clean", "ac|aero|CLmax_cruise")],
        )
        self.add_subsystem(
            "takeoff_flaps",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|takeoff_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                # Promoted to the clean maximum's name, which is how the increment is
                # applied to the value the component above just computed.
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_TO")],
        )
