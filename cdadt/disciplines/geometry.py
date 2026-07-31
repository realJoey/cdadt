"""The geometry discipline: the shape of the airframe, minus the empennage."""

from __future__ import annotations

from typing import ClassVar

from cdadt.disciplines.base import Discipline
from cdadt.parameters import Response

__all__ = ["Geometry"]


class Geometry(Discipline):
    """Wing planform, fuselage, nacelles and landing gear.

    Owns the shape parameters every other domain is a function of: the wing planform that sets
    both the lift and the wing weight, the fuselage that sets both the wetted area and the tail
    lever arm, the nacelles, and the gear.

    The empennage is *not* here. The black box sizes the horizontal and vertical stabilizers
    from tail volume coefficients rather than taking their areas as inputs, so their shape
    parameters and their computed areas belong together in
    :class:`~cdadt.disciplines.stability.Stability`.

    This class computes nothing. The mean aerodynamic chord and the wetted areas it reports are
    computed inside the black box; this discipline records that they are geometry, where to
    read them, and in what units.

    Examples
    --------
    >>> Geometry.owns("ac|geom|wing|S_ref")
    True
    >>> Geometry.owns("ac|geom|hstab|AR")
    False
    """

    discipline_name: ClassVar[str] = "geometry"
    description: ClassVar[str] = "Wing planform, fuselage, nacelles and landing gear"

    owned_patterns: ClassVar[tuple[str, ...]] = (
        "ac|geom|wing|*",
        "ac|geom|fuselage|*",
        "ac|geom|nacelle|*",
        "ac|geom|maingear|*",
        "ac|geom|nosegear|*",
    )

    reported: ClassVar[tuple[Response, ...]] = (
        Response(
            "wing_MAC",
            "ac|geom|wing|MAC",
            "m",
            "Wing mean aerodynamic chord, computed by the box from a trapezoidal planform",
        ),
        Response(
            "fuselage_wetted_area",
            "ac|geom|fuselage|S_wet",
            "m**2",
            "Fuselage wetted area, computed by the box as a cylinder",
        ),
        Response(
            "nacelle_wetted_area",
            "ac|geom|nacelle|S_wet",
            "m**2",
            "Nacelle wetted area, computed by the box as a cylinder",
        ),
        Response(
            "tail_lever_arm",
            "tail_lever_arm_estimate.c4_to_wing_c4",
            "m",
            "Wing quarter chord to tail quarter chord, estimated by the box as half the fuselage length",
            optional=True,
        ),
        Response(
            "wing_span",
            "wing_span.span",
            "m",
            "Wing span, from area and aspect ratio. Optional because it depends on which analysis "
            "group a case file names: cdadt's composes OpenConcept's WingSpan, and OpenConcept's "
            "own B738 sizing group does not",
            optional=True,
        ),
    )
