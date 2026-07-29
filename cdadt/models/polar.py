"""The parabolic drag polar: the model that proves the machinery before the physics changes.

.. math::

   C_D = C_{D_0} + \\frac{C_L^2}{\\pi e A\\!R}

This is deliberately the same equation OpenConcept's :class:`PolarDrag` evaluates, and that is
the point of it. Everything around a loads model in cdadt is new -- the abstraction, the aircraft
model, the analysis group that installs it -- and all of it has to be shown faithful before a
genuinely different aerodynamic model rides on top. A model that reproduces the reference exactly
turns "the answer changed" into evidence about one thing at a time.

It is not a stub. It is the classical polar, with analytic derivatives, and it remains the
cheapest useful model in the framework: :class:`~cdadt.adapter.avl.OpenAVLLoads` computes the
span efficiency this model takes as an input.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from cdadt.models.coefficients import AeroCoefficients
from cdadt.models.loads import AerodynamicLoads, FlightCondition, LoadsError, Planform

__all__ = ["PolarLoads"]


class PolarLoads(AerodynamicLoads):
    """Drag from a parabolic polar, given a zero-lift drag coefficient and a span efficiency.

    Parameters
    ----------
    span_efficiency : float
        Oswald span efficiency ``e``. In the shipped case this comes from ``ac|aero|polar|e``;
        :class:`~cdadt.adapter.avl.OpenAVLLoads` computes it from the lattice instead.
    zero_lift_drag : array_like, optional
        ``CD0``, either one value or one per flight point. Default 0, which makes this a pure
        induced-drag model -- useful on its own, because the induced drag is the part that
        depends on the planform being optimized.

    Raises
    ------
    LoadsError
        If the span efficiency is not positive. A zero or negative ``e`` is a division by zero
        dressed as a configuration value.

    Examples
    --------
    >>> from cdadt.models.planform import TrapezoidalPlanform
    >>> loads = PolarLoads(span_efficiency=0.82, zero_lift_drag=0.02)
    >>> wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45)
    >>> condition = FlightCondition(CL=0.5, mach=0.78)
    >>> round(float(loads.coefficients(condition, wing).CD), 6)
    0.030246
    """

    model_name: ClassVar[str] = "parabolic_polar"

    requires: ClassVar[tuple[str, ...]] = ("ac|aero|polar|e", "ac|geom|wing|S_ref", "ac|geom|wing|AR")

    __slots__ = ("_span_efficiency", "_zero_lift_drag")

    def __init__(self, span_efficiency: float, zero_lift_drag: object = 0.0) -> None:
        if span_efficiency <= 0.0:
            raise LoadsError(
                f"The span efficiency must be positive; got {span_efficiency!r}. It divides the "
                f"induced drag term, so zero is not 'no induced drag' but a division by zero."
            )
        self._span_efficiency = float(span_efficiency)
        self._zero_lift_drag = np.atleast_1d(np.asarray(zero_lift_drag, dtype=float))

    @classmethod
    def build(cls, *, planform, span_efficiency, zero_lift_drag):
        """Build from the span efficiency and zero-lift drag; the wing's shape is not used.

        A parabolic polar sees the wing only through its aspect ratio, which reaches it via the
        planform passed to :meth:`coefficients`. The shape -- sweep, taper, the sections -- makes
        no difference to this model, and saying so here is more honest than accepting it and
        quietly ignoring it.
        """
        return cls(span_efficiency=span_efficiency, zero_lift_drag=zero_lift_drag)

    @property
    def span_efficiency(self) -> float:
        """Oswald span efficiency."""
        return self._span_efficiency

    @property
    def zero_lift_drag(self) -> np.ndarray:
        """Zero-lift drag coefficient, as given."""
        return self._zero_lift_drag

    def induced_drag_factor(self, planform: Planform) -> float:
        """Return ``K`` in ``CD = CD0 + K CL^2``, which is ``1 / (pi e AR)``."""
        return 1.0 / (np.pi * self._span_efficiency * planform.aspect_ratio)

    def coefficients(self, condition: FlightCondition, planform: Planform) -> AeroCoefficients:
        """Return the coefficients at every point of ``condition``.

        Only lift and drag are non-zero. A parabolic polar carries no information about
        sideforce or moments, and reporting a fabricated zero moment would be a different claim
        from reporting that the model does not produce one -- which is what
        :attr:`~cdadt.models.coefficients.AeroCoefficients.lateral_directional` says.
        """
        lift = condition.CL
        drag = self._zero_lift_drag + self.induced_drag_factor(planform) * lift**2
        return AeroCoefficients(CL=lift, CD=drag)

    def drag_gradients(self, condition: FlightCondition, planform: Planform) -> dict[str, np.ndarray]:
        """Return the analytic derivatives of ``CD``, for the component that installs this model.

        Written here rather than in the OpenMDAO wrapper because they belong to the equation, and
        an optimizer steps on them. Keys are the names the wrapper declares partials against.
        """
        lift = condition.CL
        factor = self.induced_drag_factor(planform)
        return {
            "CL": 2.0 * factor * lift,
            "CD0": np.ones_like(lift),
            "e": -factor * lift**2 / self._span_efficiency,
            "AR": -factor * lift**2 / planform.aspect_ratio,
        }
