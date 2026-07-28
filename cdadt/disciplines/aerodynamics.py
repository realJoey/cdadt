"""The aerodynamics discipline: the drag and high-lift parameters, and what they produce."""

from __future__ import annotations

from typing import ClassVar

from cdadt.disciplines.base import Discipline
from cdadt.parameters import Response

__all__ = ["Aerodynamics"]


class Aerodynamics(Discipline):
    """Span efficiency, the airfoil, the flap setting and the speed envelope.

    Owns the ``ac|aero|`` parameters: the Oswald span efficiency factor that closes the drag
    polar, the airfoil section maximum lift coefficient and the takeoff flap deflection that
    together set the two maximum lift coefficients, the maximum operating Mach number, and the
    landing stall speed.

    Every drag number in a cdadt run is computed inside the black box, at every node of every
    phase, by OpenConcept's component-by-component parasite drag buildup and its parabolic
    polar. This class writes the inputs to that buildup and reads the two lift coefficients back
    out; it contains no aerodynamic model of its own and no aerodynamic constant.

    Notes
    -----
    The drag history itself is per-phase and vector-valued, so it is not reported here. It is
    read directly from the box by name -- ``mission.cruise.drag`` and its siblings -- when a run
    is plotted or exported.
    """

    discipline_name: ClassVar[str] = "aerodynamics"
    description: ClassVar[str] = "Span efficiency, airfoil and flap settings, speed envelope"

    owned_patterns: ClassVar[tuple[str, ...]] = ("ac|aero|*",)

    reported: ClassVar[tuple[Response, ...]] = (
        Response(
            "CLmax_cruise",
            "ac|aero|CLmax_cruise",
            None,
            "Clean maximum lift coefficient, computed by the box from the airfoil and sweep",
        ),
        Response(
            "CLmax_takeoff",
            "ac|aero|CLmax_TO",
            None,
            "Maximum lift coefficient with takeoff flaps, computed by the box",
        ),
    )
