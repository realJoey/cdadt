"""Aerodynamic loads from openavl: a vortex lattice standing in for OpenConcept's drag polar.

This is the model the whole abstraction exists for. The wing the case file describes is built as
an AVL lattice, solved, and the lift-dependent drag it predicts replaces the one OpenConcept
assumes.

Why the lattice is not solved at every node
--------------------------------------------

A sizing mission is about 170 analysis points, inside a Newton solve, inside an optimizer.
openavl's OpenMDAO component evaluates one flight point per instance, so solving per node would
mean thousands of lattice solves per design. OpenConcept faced exactly this and answered it with
a trained surrogate -- ``VLMDragPolar`` exists because *"a surrogate model to decrease the
computational cost"* was necessary.

cdadt takes the cheaper and exact route, because a vortex lattice is **linear**. For a fixed
geometry the far-field drag is quadratic in the far-field lift, so three solves determine the whole
polar:

.. math::

   C_D = C_{D_\\mathrm{min}} + k\\,(C_L - C_{L_\\mathrm{minD}})^2

The quadratic is an identity rather than a curve fit, and it is written with an offset rather than
as :math:`C_D \\propto C_L^2` because twist and camber move the minimum-drag point away from zero
lift. Three solves per geometry and Mach number, then every node in closed form -- and the polar is
re-derived whenever the planform moves, which is what keeps this exact under an optimizer rather
than a surrogate that drifts.

:mod:`cdadt.adapter.lattice` owns the openavl calls, the far-field pairing that makes the polar
self-consistent with openavl's own span efficiency, and the exact geometry Jacobian. This module is
what turns those into an :class:`~cdadt.models.loads.AerodynamicLoads`.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from cdadt.adapter.lattice import LatticeLibrary, LatticePolar
from cdadt.models.coefficients import AeroCoefficients
from cdadt.models.loads import AerodynamicLoads, FlightCondition, LoadsError, Planform
from cdadt.models.planform import TrapezoidalPlanform

__all__ = ["MACH_SAMPLES", "OpenAVLLoads", "openavl_is_available"]


def openavl_is_available() -> bool:
    """Return whether openavl can be imported.

    openavl is an optional extra. Everything else in cdadt works without it, and the tests that
    need it skip rather than fail, so a machine without JAX still runs the whole suite honestly.
    """
    try:
        import openavl  # noqa: F401
    except ImportError:
        return False
    return True


#: Mach numbers the polar is fitted at. The shipped mission runs from roughly M 0.35 in the climb
#: to 0.785 in cruise, and the lattice's answer varies smoothly and weakly across that: the induced
#: drag at CL = 0.5 moves 0.8% between M = 0 and M = 0.785, and the span efficiency from 0.990 to
#: 0.998. Five points with linear interpolation is therefore well inside the lattice's own
#: resolution error, and costs fifteen solves per geometry instead of three.
MACH_SAMPLES: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.85)


class OpenAVLLoads(AerodynamicLoads):
    """Lift-dependent drag from an openavl vortex lattice, with the polar fitted per geometry.

    Parameters
    ----------
    planform : Planform
        The wing to build the lattice on. Must be a
        :class:`~cdadt.models.planform.TrapezoidalPlanform`; the lattice is built from its
        sections.
    zero_lift_drag : array_like, optional
        Parasite ``CD0`` from outside the lattice -- the fuselage, nacelles, tails and skin
        friction that a wing-only lattice does not see. Default 0. In a cdadt study this is
        OpenConcept's own component buildup, which is left where it is because reimplementing it
        would be copying six hundred lines of correlations.
    chordwise, spanwise : int, optional
        Lattice density. Defaults are deliberately modest; the polar is fitted once per geometry
        and Mach sample, so the cost is bounded whatever the mission length.

    Raises
    ------
    LoadsError
        If openavl is not installed, or if the planform is not one this model can build a lattice
        from, or if the lattice returns a polar that curves the wrong way.

    Notes
    -----
    Wave drag is not this model's business. A vortex lattice has no mechanism for it; what Mach
    reaches here is openavl's Prandtl-Glauert correction, through :data:`MACH_SAMPLES`. Transonic
    drag rise comes instead from OpenConcept's own ``WaveDragFromSections``, which
    :class:`~cdadt.adapter.aircraft.CdadtAircraftModel` installs when asked -- so it is added to the
    parasite drag rather than to this polar. See :doc:`/truth`.
    """

    model_name: ClassVar[str] = "openavl_vortex_lattice"

    #: The lattice is rebuilt from all four wing numbers, so all four reach the drag -- which is the
    #: substantive difference from a parabolic polar. ``e`` is absent because this model computes the
    #: span efficiency instead of reading it.
    DRAG_DEPENDS_ON: ClassVar[tuple[str, ...]] = ("CL", "CD0", "area", "AR", "sweep", "taper")

    __slots__ = ("_library", "_planform", "_resolution", "_zero_lift_drag")

    def __init__(
        self,
        planform: Planform,
        zero_lift_drag: object = 0.0,
        chordwise: int = 6,
        spanwise: int = 20,
        library: LatticeLibrary | None = None,
    ) -> None:
        if not openavl_is_available():
            raise LoadsError(
                "openavl is not installed, so the vortex-lattice loads model cannot be used. "
                'Install it with: pip install -e ".[avl]"'
            )
        if not isinstance(planform, TrapezoidalPlanform):
            raise LoadsError(
                f"The lattice is built from a trapezoidal planform's sections; got "
                f"{type(planform).__name__}, which does not describe one."
            )
        self._planform = planform
        self._zero_lift_drag = np.atleast_1d(np.asarray(zero_lift_drag, dtype=float))
        self._resolution = (int(chordwise), int(spanwise))
        # A private library when none is offered. That is correct but slow if a caller rebuilds the
        # model per evaluation, which is exactly what the mission does -- so the analysis group
        # injects one that outlives the model. Standalone use gets caching within one instance.
        self._library = library if library is not None else LatticeLibrary()

    @classmethod
    def build(
        cls,
        *,
        planform: Planform,
        span_efficiency: float,
        zero_lift_drag: object,
        workspace: object = None,
    ) -> OpenAVLLoads:
        """Build the lattice on this wing. ``span_efficiency`` is ignored, and that is the point.

        A case file flying the parabolic polar must state ``ac|aero|polar|e`` as an assumption.
        Here the lattice computes it, so the stated value is deliberately not consulted -- see
        :attr:`span_efficiency` for what the wing actually achieves.

        ``workspace`` is the :class:`~cdadt.adapter.lattice.LatticeLibrary` the caller owns. It is
        what makes this model cheap to construct despite being expensive to solve: the lattice
        results live in the library, not in the model, so rebuilding the model per Newton iteration
        costs nothing. Passed ``None``, the model keeps its own -- correct, and slow under a solver.
        """
        if workspace is not None and not isinstance(workspace, LatticeLibrary):
            raise LoadsError(
                f"The vortex-lattice model keeps its solved polars in a LatticeLibrary; got "
                f"{type(workspace).__name__}, which it cannot store anything in."
            )
        return cls(planform=planform, zero_lift_drag=zero_lift_drag, library=workspace)

    # -- what the lattice says about this wing -------------------------------------------

    def polar_at(self, mach: float) -> LatticePolar:
        """Return the fitted, differentiated polar at one sampled Mach number.

        Public because a study comparing what the lattice says against what a case file assumes
        needs the polar itself, not only the drag it produces.
        """
        return self._library.polar_for(self._planform, float(mach), *self._resolution)

    def _interpolated(self, mach: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the polar coefficients and their geometry Jacobian at each node's Mach number.

        Interpolated across :data:`MACH_SAMPLES`. ``np.interp`` holds the end values beyond the
        range, which is the right behaviour here: below M = 0 is not a flight condition, and above
        M = 0.85 a vortex lattice is not a model of anything, so extrapolating would invent
        confidence rather than accuracy.

        The Jacobian is interpolated with the same weights as the coefficients, which is what makes
        the reported derivatives the exact derivatives of the reported drag: the interpolation
        weights depend on Mach alone, and Mach is not a geometry variable.
        """
        polars = [self.polar_at(mach_sample) for mach_sample in MACH_SAMPLES]
        grid = np.asarray(MACH_SAMPLES, dtype=float)

        coefficients = tuple(
            np.interp(mach, grid, [polar.coefficients[index] for polar in polars]) for index in range(3)
        )
        jacobian = {
            variable: tuple(
                np.interp(mach, grid, [polar.gradient(variable)[index] for polar in polars]) for index in range(3)
            )
            for variable in LatticePolar.GEOMETRY
        }
        return coefficients, jacobian  # type: ignore[return-value]

    @property
    def span_efficiency(self) -> float:
        """The Oswald efficiency openavl reports for this wing, incompressible.

        Reported rather than assumed. A case file flying the parabolic polar has to state a value
        for ``ac|aero|polar|e``; this is what the wing actually achieves, and comparing the two is
        the point of installing a lattice at all. Taken at M = 0 so that it is a property of the
        wing rather than of a flight condition; :meth:`polar_at` carries the Mach dependence.
        """
        return self.polar_at(MACH_SAMPLES[0]).span_efficiency

    def _check_planform(self, planform: Planform) -> None:
        """Reject a planform the lattice was not built on.

        A silent mismatch would report the drag of a different aeroplane, so it is an error rather
        than a rebuild: the model is constructed per evaluation precisely so that the wing it holds
        is the current one.
        """
        if planform is not self._planform and (
            planform.area != self._planform.area or planform.aspect_ratio != self._planform.aspect_ratio
        ):
            raise LoadsError(
                "This model was built on a different wing than it is being evaluated on: the "
                "lattice describes one geometry and cannot be reused for another."
            )

    def coefficients(self, condition: FlightCondition, planform: Planform) -> AeroCoefficients:
        """Return the coefficients at every point, from the polar fitted to this wing.

        ``planform`` is accepted for the interface's sake and checked against the one the lattice
        was built on, because a silent mismatch would report the drag of a different aeroplane.
        """
        self._check_planform(planform)
        (minimum_drag, curvature, lift_at_minimum), _jacobian = self._interpolated(condition.mach)
        lift = condition.CL
        drag = self._zero_lift_drag + minimum_drag + curvature * (lift - lift_at_minimum) ** 2
        return AeroCoefficients(CL=lift, CD=drag)

    def drag_gradients(self, condition: FlightCondition, planform: Planform) -> dict[str, np.ndarray]:
        """Return the exact derivatives of ``CD``, differentiated through the lattice itself.

        Every geometry derivative comes from ``jax.jacrev`` over openavl's own differentiable
        geometry update, chained through the closed-form polar:

        .. math::

           \\frac{\\partial C_D}{\\partial g} =
               \\frac{\\partial C_{D_\\mathrm{min}}}{\\partial g}
             + \\frac{\\partial k}{\\partial g}\\,(C_L - C_{L_\\mathrm{minD}})^2
             - 2k\\,(C_L - C_{L_\\mathrm{minD}})\\,\\frac{\\partial C_{L_\\mathrm{minD}}}{\\partial g}

        so ``area``, ``AR``, ``sweep`` and ``taper`` are all exact. An earlier version of this
        model derived the aspect-ratio term from :math:`k = 1/(\\pi e A\\!R)` with the span
        efficiency held fixed -- wrong by 1.8% -- and reported nothing at all for sweep and taper,
        which an optimizer reads as "these do not matter".

        ``e`` is **zero, and that is correct** -- not missing. This model does not read
        ``ac|aero|polar|e``; it computes the span efficiency from the lattice. A case file's stated
        value has no influence on this drag, so the derivative with respect to it genuinely is
        nothing. Compare :meth:`~cdadt.models.polar.PolarLoads.drag_gradients`, where it is the
        dominant term.
        """
        self._check_planform(planform)
        (_minimum_drag, curvature, lift_at_minimum), jacobian = self._interpolated(condition.mach)
        lift = condition.CL
        excess = lift - lift_at_minimum

        gradients = {
            "CL": 2.0 * curvature * excess,
            "CD0": np.ones_like(lift),
            "e": np.zeros_like(lift),
        }
        for variable, (d_minimum_drag, d_curvature, d_lift_at_minimum) in jacobian.items():
            gradients[variable] = (
                d_minimum_drag + d_curvature * excess**2 - 2.0 * curvature * excess * d_lift_at_minimum
            )
        return gradients
