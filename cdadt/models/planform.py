"""The trapezoidal wing a case file describes, and the lifting surface it implies.

A case file declares a wing the way a designer states one: reference area, aspect ratio, quarter
chord sweep and taper ratio. A vortex-lattice method needs something else entirely -- the
leading-edge coordinates and chord of every section. This module is the translation, and it is
cdadt's own geometry rather than anyone else's physics.

The same four numbers serve both models cdadt ships. :class:`~cdadt.models.polar.PolarLoads`
reads only :attr:`~TrapezoidalPlanform.area` and :attr:`~TrapezoidalPlanform.aspect_ratio`;
:class:`~cdadt.adapter.avl.OpenAVLLoads` builds a surface from :meth:`TrapezoidalPlanform.sections`.
That is the point of :class:`~cdadt.models.loads.Planform` being abstract in what it exposes.
"""

from __future__ import annotations

import numpy as np

from cdadt.models.loads import Planform

__all__ = ["TrapezoidalPlanform", "WingSection"]


class WingSection:
    """One spanwise station of a lifting surface: where its leading edge is, and its chord.

    The description a vortex-lattice method works in. Coordinates are in the aircraft's own
    frame, metres, with ``x`` aft, ``y`` to starboard and ``z`` up.
    """

    __slots__ = ("_chord", "_x", "_y", "_z")

    def __init__(self, x: float, y: float, z: float, chord: float) -> None:
        if chord <= 0.0:
            raise ValueError(f"A wing section needs a positive chord; got {chord!r}.")
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._chord = float(chord)

    @property
    def x(self) -> float:
        """Leading-edge station, metres aft."""
        return self._x

    @property
    def y(self) -> float:
        """Spanwise station, metres from the centreline."""
        return self._y

    @property
    def z(self) -> float:
        """Leading-edge height, metres."""
        return self._z

    @property
    def chord(self) -> float:
        """Section chord, metres."""
        return self._chord

    def __repr__(self) -> str:
        """Return a representation naming the station and chord."""
        return f"WingSection(y={self._y:.3f}, chord={self._chord:.3f})"


class TrapezoidalPlanform(Planform):
    """A straight-tapered wing built from the four numbers a case file declares.

    Parameters
    ----------
    area : float
        Reference area, square metres.
    aspect_ratio : float
        Aspect ratio, span squared over area.
    sweep : float, optional
        Quarter-chord sweep, **degrees**. Default 0. Degrees because that is the unit the case
        file and 14 CFR both use; it is converted once, here.
    taper : float, optional
        Tip chord over root chord. Default 1, an untapered wing. Values above 1 describe an
        inverse-taper wing, which is unusual but perfectly buildable, so they are allowed.

    Raises
    ------
    ValueError
        If the area, aspect ratio or taper is not positive. Zero taper is a wing with no tip
        chord, which no lattice can be built on.

        Note what is *not* rejected: a taper above 1. An earlier version capped it there, on the
        reasoning that a tapered wing narrows outboard. That made the class non-differentiable at
        exactly the bound an optimizer is most likely to sit on -- a finite difference at
        ``taper = 1`` steps to 1.000001 and threw. A constraint that breaks the derivative at its
        own boundary belongs in a case file's ``optimize`` band, not in a geometry class.

    Examples
    --------
    >>> wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    >>> round(wing.span, 3)
    34.31
    >>> round(wing.root_chord, 3)
    6.267
    """

    __slots__ = ("_area", "_aspect_ratio", "_sweep_deg", "_taper")

    def __init__(self, area: float, aspect_ratio: float, sweep: float = 0.0, taper: float = 1.0) -> None:
        if area <= 0.0:
            raise ValueError(f"A wing needs a positive reference area; got {area!r}.")
        if aspect_ratio <= 0.0:
            raise ValueError(f"A wing needs a positive aspect ratio; got {aspect_ratio!r}.")
        if taper <= 0.0:
            raise ValueError(
                f"Taper ratio is the tip chord over the root chord and must be positive; got {taper!r}. "
                f"Zero leaves no tip chord to build a section on."
            )
        self._area = float(area)
        self._aspect_ratio = float(aspect_ratio)
        self._sweep_deg = float(sweep)
        self._taper = float(taper)

    # -- what every planform exposes -----------------------------------------------------

    @property
    def area(self) -> float:
        """Reference wing area, square metres."""
        return self._area

    @property
    def aspect_ratio(self) -> float:
        """Wing aspect ratio."""
        return self._aspect_ratio

    # -- what a trapezoid adds -----------------------------------------------------------

    @property
    def sweep(self) -> float:
        """Quarter-chord sweep, degrees."""
        return self._sweep_deg

    @property
    def taper(self) -> float:
        """Tip chord over root chord."""
        return self._taper

    @property
    def span(self) -> float:
        """Wing span, metres."""
        return float(np.sqrt(self._aspect_ratio * self._area))

    @property
    def root_chord(self) -> float:
        """Centreline chord, metres.

        From ``S = b * c_root * (1 + taper) / 2`` for a straight-tapered wing.
        """
        return 2.0 * self._area / (self.span * (1.0 + self._taper))

    @property
    def tip_chord(self) -> float:
        """Chord at the tip, metres."""
        return self.root_chord * self._taper

    @property
    def mean_aerodynamic_chord(self) -> float:
        """Mean aerodynamic chord, metres.

        The standard trapezoidal result, ``(2/3) c_root (1 + t + t^2) / (1 + t)``.
        """
        taper = self._taper
        return (2.0 / 3.0) * self.root_chord * (1.0 + taper + taper**2) / (1.0 + taper)

    def sections(self) -> tuple[WingSection, ...]:
        """Return the root and tip sections of the starboard semi-span.

        Two sections define a trapezoid completely, and a lattice method panels between them. The
        tip leading edge is placed so that the **quarter chord** carries the stated sweep, which
        is what ``ac|geom|wing|c4sweep`` means -- sweeping the leading edge instead would give a
        different wing for the same case file.
        """
        semi_span = 0.5 * self.span
        root_chord = self.root_chord
        tip_chord = self.tip_chord

        quarter_chord_offset = semi_span * np.tan(np.radians(self._sweep_deg))
        # Convert quarter-chord sweep to a leading-edge offset: the tip quarter chord sits at
        # the swept station, so its leading edge is a quarter of its own chord ahead of it.
        tip_leading_edge = quarter_chord_offset + 0.25 * root_chord - 0.25 * tip_chord

        return (
            WingSection(x=0.0, y=0.0, z=0.0, chord=root_chord),
            WingSection(x=tip_leading_edge, y=semi_span, z=0.0, chord=tip_chord),
        )

    def __repr__(self) -> str:
        """Return a representation naming the planform."""
        return (
            f"TrapezoidalPlanform(area={self._area:.4g}, aspect_ratio={self._aspect_ratio:.4g}, "
            f"sweep={self._sweep_deg:.4g}, taper={self._taper:.4g})"
        )
