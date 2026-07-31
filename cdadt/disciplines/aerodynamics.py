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

    This class writes the inputs to whatever drag model the study installed and reads the two lift
    coefficients back out. **It contains no aerodynamic model of its own and no aerodynamic
    constant** -- that invariant still holds, and it is what keeps a discipline an interface rather
    than a second place physics can live.

    What has changed underneath it is which model those inputs reach. On the default path every drag
    number is computed inside the black box by OpenConcept's component-by-component parasite buildup
    and its parabolic polar. A study may instead name one of cdadt's own -- see
    :mod:`cdadt.models.loads` and :doc:`/aerodynamics` -- in which case the induced drag is computed
    by a vortex lattice built on this wing and ``ac|aero|polar|e`` becomes an unused assumption,
    because the lattice reports the span efficiency rather than reading it.

    Either way the ownership is unchanged: this class owns the ``ac|aero|`` *parameters*, and the
    model that consumes them is chosen by the case file, not by this discipline.

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
