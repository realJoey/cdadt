"""The stability discipline: empennage sizing by tail volume coefficient."""

from __future__ import annotations

from typing import ClassVar

from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing

from cdadt.disciplines.base import AircraftDiscipline

__all__ = ["Stability"]


class Stability(AircraftDiscipline):
    """Horizontal and vertical stabilizer areas from tail volume coefficients.

    Raymer's conceptual-design tail sizing (1992 edition, Section 6.4, Equations 6.26 and
    6.27):

    .. math::

       S_\\mathrm{ht} = \\frac{C_\\mathrm{ht}\\,\\overline{c}\\,S_\\mathrm{ref}}{l_\\mathrm{ht}},
       \\qquad
       S_\\mathrm{vt} = \\frac{C_\\mathrm{vt}\\,\\sqrt{\\mathrm{AR}}\\,S_\\mathrm{ref}^{3/2}}{l_\\mathrm{vt}}

    Sizing the tails rather than fixing their areas is what makes a wing-planform design
    variable honest: growing the wing grows the empennage that stabilizes it, and the empty
    weight follows. Fixed tail areas would let the optimizer buy wing area for free.

    Inputs
    ------
    ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|MAC : float
        Wing planform and mean aerodynamic chord (scalar).
    ac|geom|hstab|c4_to_wing_c4, ac|geom|vstab|c4_to_wing_c4 : float
        Tail lever arms from :class:`~cdadt.disciplines.geometry.Geometry` (scalar, m).

    Outputs
    -------
    ac|geom|hstab|S_ref, ac|geom|vstab|S_ref : float
        Stabilizer reference areas (scalar, m**2).

    Notes
    -----
    The volume coefficients are class attributes carrying Raymer's Table 6.4 jet-transport
    values. Subclass to size a different class of aircraft -- twin turboprop is 0.9 and 0.08.
    """

    discipline_name: ClassVar[str] = "stability"

    #: Horizontal tail volume coefficient. Raymer 1992 Table 6.4, jet transport.
    horizontal_tail_volume_coefficient: ClassVar[float] = 1.00

    #: Vertical tail volume coefficient. Raymer 1992 Table 6.4, jet transport.
    vertical_tail_volume_coefficient: ClassVar[float] = 0.09

    def setup(self) -> None:
        """Add the two tail-sizing components."""
        self.add_subsystem(
            "hstab_area",
            HStabVolumeCoefficientSizing(C_ht=self.horizontal_tail_volume_coefficient),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|MAC", "ac|geom|hstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|hstab|S_ref"],
        )
        self.add_subsystem(
            "vstab_area",
            VStabVolumeCoefficientSizing(C_vt=self.vertical_tail_volume_coefficient),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|geom|vstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|vstab|S_ref"],
        )
