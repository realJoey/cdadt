"""The wing's sections, published as OpenMDAO variables so OpenConcept's components can read them.

:class:`~cdadt.models.planform.TrapezoidalPlanform` already turns the four numbers a case file
declares into spanwise stations and chords. This is that translation as a component, because one of
the things cdadt installs -- OpenConcept's own ``WaveDragFromSections`` -- asks for the wing
*section by section* rather than as an area and an aspect ratio.

It is in :mod:`cdadt.adapter` rather than in :mod:`cdadt.models` for the usual reason: a model is
plain Python and a component is OpenMDAO. It computes nothing of its own -- the formulas live once,
in the planform.

Section ordering
----------------

``WaveDragFromSections`` wants sections *"starting with the outboard section (wing tip) at the MOST
NEGATIVE y value and moving inboard"*, and it wants ``y_sec`` for every section **except** the root,
because the root is always zero. Those two conventions are its own, they are easy to get backwards,
and getting them backwards would silently describe a wing with the tip chord at the root. So they
are honoured here explicitly and checked by a test that reads the tip chord back out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import openmdao.api as om

from cdadt.models.planform import TrapezoidalPlanform

__all__ = ["WingSectionsComp"]


class WingSectionsComp(om.ExplicitComponent):
    """Publish a trapezoidal wing's spanwise stations and chords from the four numbers it is given.

    Inputs are ``ac|geom|wing|S_ref``, ``ac|geom|wing|AR``, ``ac|geom|wing|taper`` and
    ``ac|geom|wing|toverc``; outputs are ``y_sec`` (the tip station, negative, one value because the
    root's is zero by convention), ``chord_sec`` (tip then root) and ``toverc_sec``.

    The thickness ratio is broadcast rather than distributed. A case file describes a trapezoidal
    wing with a single ``ac|geom|wing|toverc``, so giving the tip a different value would be
    inventing geometry the study never stated. It is published here rather than assembled with an
    :class:`~openmdao.components.exec_comp.ExecComp` in the aircraft model because a section
    quantity belongs with the other section quantities, and because one component with analytic
    partials is easier to check than two.

    Derivatives are analytic and closed-form. Writing the root chord as

    .. math::

       c_\\mathrm{root} = \\frac{2\\sqrt{S}}{\\sqrt{A\\!R}\\,(1 + \\lambda)}

    makes every derivative a multiple of the quantity itself, which is why they are one-liners
    rather than an application of the quotient rule to :math:`2S / (b(1+\\lambda))`.
    """

    def initialize(self) -> None:
        """Declare nothing: a wing has one shape however many nodes a phase has."""

    def setup(self) -> None:
        """Declare the three wing numbers read and the two section arrays published."""
        self.add_input("ac|geom|wing|S_ref", shape=(1,), units="m**2")
        self.add_input("ac|geom|wing|AR", shape=(1,))
        self.add_input("ac|geom|wing|taper", shape=(1,))
        self.add_input("ac|geom|wing|toverc", shape=(1,))

        # One station, not two: the root is at y = 0 by the consumer's convention.
        self.add_output("y_sec", shape=(1,), units="m")
        self.add_output("chord_sec", shape=(2,), units="m")
        self.add_output("toverc_sec", shape=(2,))

        self.declare_partials("y_sec", ["ac|geom|wing|S_ref", "ac|geom|wing|AR"])
        self.declare_partials("chord_sec", ["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|geom|wing|taper"])
        # Constant, so it is declared once here and never touched in compute_partials.
        self.declare_partials("toverc_sec", "ac|geom|wing|toverc", val=np.ones((2, 1)))

    def _wing(self, inputs: Any) -> tuple[dict[str, object], float, float, float]:
        """Return the planform's derived dimensions and the three numbers they came from."""
        area = float(inputs["ac|geom|wing|S_ref"][0])
        aspect_ratio = float(inputs["ac|geom|wing|AR"][0])
        taper = float(inputs["ac|geom|wing|taper"][0])
        return TrapezoidalPlanform.geometry(area, aspect_ratio, 0.0, taper), area, aspect_ratio, taper

    def compute(self, inputs: Any, outputs: Any) -> None:
        """Publish the tip station and the two chords, outboard first."""
        wing, _area, _aspect_ratio, _taper = self._wing(inputs)

        # Negative, and outboard first: the consumer's convention, not a sign error.
        outputs["y_sec"] = -float(wing["semi_span"])
        outputs["chord_sec"] = np.array([float(wing["tip_chord"]), float(wing["root_chord"])])
        outputs["toverc_sec"] = np.full(2, float(inputs["ac|geom|wing|toverc"][0]))

    def compute_partials(self, inputs: Any, partials: Any) -> None:
        """Publish the closed-form derivatives of the stations and chords."""
        wing, area, aspect_ratio, taper = self._wing(inputs)
        semi_span = float(wing["semi_span"])
        root_chord = float(wing["root_chord"])
        tip_chord = float(wing["tip_chord"])

        partials["y_sec", "ac|geom|wing|S_ref"] = -semi_span / (2.0 * area)
        partials["y_sec", "ac|geom|wing|AR"] = -semi_span / (2.0 * aspect_ratio)

        partials["chord_sec", "ac|geom|wing|S_ref"] = np.array([[tip_chord], [root_chord]]) / (2.0 * area)
        partials["chord_sec", "ac|geom|wing|AR"] = -np.array([[tip_chord], [root_chord]]) / (2.0 * aspect_ratio)
        # The tip chord grows with taper and the root chord shrinks, by the same amount and with
        # opposite signs, because the area they share is fixed.
        partials["chord_sec", "ac|geom|wing|taper"] = np.array([[1.0], [-1.0]]) * root_chord / (1.0 + taper)
