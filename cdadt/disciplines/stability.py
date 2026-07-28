"""The stability discipline: the empennage, sized by tail volume coefficient."""

from __future__ import annotations

from typing import ClassVar

from cdadt.disciplines.base import Discipline
from cdadt.parameters import Response

__all__ = ["Stability"]


class Stability(Discipline):
    """Horizontal and vertical stabilizer shape, and the areas the box sizes from it.

    Owns the empennage shape parameters -- aspect ratio, quarter-chord sweep, taper and
    thickness ratio for both surfaces -- and reports the two reference areas.

    The areas are *outputs*, not inputs, which is why the empennage is a discipline of its own
    rather than part of :class:`~cdadt.disciplines.geometry.Geometry`. The black box sizes both
    surfaces by the tail volume coefficient method: the horizontal tail from wing area, mean
    aerodynamic chord and lever arm, the vertical tail from wing area, span and lever arm. The
    volume coefficients themselves (1.00 horizontal, 0.09 vertical, Raymer Table 6.4 for jet
    transports) are *options* of the components inside the box, not inputs to it, so they cannot
    be set from cdadt and are documented as fixed by the box. See :doc:`/blackbox`.

    That also means this discipline models no stability physics whatsoever -- no static margin,
    no centre of gravity, no minimum control speed. It is named for the domain the tail volume
    coefficient method belongs to, and :doc:`/validation` states the gap explicitly.
    """

    discipline_name: ClassVar[str] = "stability"
    description: ClassVar[str] = "Empennage shape, and the tail areas the box sizes by volume coefficient"

    owned_patterns: ClassVar[tuple[str, ...]] = ("ac|geom|hstab|*", "ac|geom|vstab|*")

    reported: ClassVar[tuple[Response, ...]] = (
        Response(
            "hstab_area",
            "ac|geom|hstab|S_ref",
            "m**2",
            "Horizontal stabilizer reference area, sized by the box from the tail volume coefficient",
        ),
        Response(
            "vstab_area",
            "ac|geom|vstab|S_ref",
            "m**2",
            "Vertical stabilizer reference area, sized by the box from the tail volume coefficient",
        ),
    )
