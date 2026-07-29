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
geometry the induced drag is quadratic in lift, so a handful of solves determines the whole polar:

.. math::

   C_D = C_{D_\\mathrm{min}} + k\\,(C_L - C_{L_\\mathrm{minD}})^2

The fit is quadratic rather than proportional because twist and camber move the minimum-drag
point away from zero lift; assuming :math:`C_D \\propto C_L^2` would silently mis-model any wing
with washout. Three solves per geometry, then every node in closed form -- and the fit is
re-derived whenever the planform moves, which is what keeps it exact under an optimizer rather
than a surrogate that drifts.

The span efficiency this implies is reported as :attr:`~OpenAVLLoads.span_efficiency`, so a study
can compare what the lattice says against the ``ac|aero|polar|e`` a case file would otherwise
have to assume.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import ClassVar

import numpy as np

from cdadt.models.coefficients import AeroCoefficients
from cdadt.models.loads import AerodynamicLoads, FlightCondition, LoadsError, Planform
from cdadt.models.planform import TrapezoidalPlanform

__all__ = ["OpenAVLLoads", "openavl_is_available", "quadratic_polar"]


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


def _lattice(
    planform: TrapezoidalPlanform,
    mach: float,
    chordwise: int,
    spanwise: int,
    aircraft_class: type,
) -> object:
    """Build the openavl aircraft for a trapezoidal wing.

    ``aircraft_class`` is passed in rather than imported here so that this function carries no
    import of its own: the dependency is optional, and only the cached solves that need it import
    it. The reference quantities are the planform's own, which is what makes the coefficients the
    lattice returns comparable with the ones the mission works in.
    """
    root, tip = planform.sections()
    aircraft = aircraft_class(
        name="cdadt wing",
        mach=mach,
        sref=planform.area,
        cref=planform.mean_aerodynamic_chord,
        bref=planform.span,
        xref=0.25 * planform.mean_aerodynamic_chord,
        yref=0.0,
        zref=0.0,
    )
    wing = aircraft.add_wing(
        "wing", n_chord=chordwise, c_space=1.0, n_span=spanwise, s_space=-2.0, symmetric=True, component=1
    )
    for section in (root, tip):
        wing.add_section(xyzle=[section.x, section.y, section.z], chord=section.chord, n_span=1, s_space=0.0)
    return aircraft


def quadratic_polar(
    lift_coefficients: Sequence[float],
    drag_coefficients: Sequence[float],
    describes: str = "this wing",
) -> tuple[float, float, float]:
    """Fit ``CD = cd_min + k (CL - cl_at_min_drag)^2`` and return those three numbers.

    Separated from the lattice solve so that the fit can be tested on its own, including the
    degenerate cases -- a lattice that returns drag falling with lift is a geometry the solver
    could not resolve, and that must be an error rather than a negative curvature propagating
    into an optimizer as an incentive to add lift.

    Raises
    ------
    LoadsError
        If the fitted curvature is not positive. Induced drag grows with lift; anything else is
        not a drag polar.
    """
    a, b, c = np.polyfit(np.asarray(lift_coefficients, dtype=float), np.asarray(drag_coefficients, dtype=float), 2)
    if a <= 0.0:
        raise LoadsError(
            f"The lattice produced a drag polar with non-positive curvature ({a:.3e}) for "
            f"{describes}. Induced drag must grow with lift, so this is a geometry the lattice "
            f"could not resolve rather than a result."
        )
    cl_at_min_drag = -b / (2.0 * a)
    cd_min = c - a * cl_at_min_drag**2
    return float(cd_min), float(a), float(cl_at_min_drag)


@lru_cache(maxsize=32)
def _fit_polar(
    area: float,
    aspect_ratio: float,
    sweep: float,
    taper: float,
    mach: float,
    chordwise: int,
    spanwise: int,
) -> tuple[float, float, float]:
    """Solve the lattice at three lift coefficients and return the fitted polar.

    Returns ``(cd_min, curvature, cl_at_min_drag)`` for
    ``CD = cd_min + curvature * (CL - cl_at_min_drag)**2``.

    Cached on the geometry because an optimizer asks for the same wing at every node of every
    phase within one design iteration, and the lattice answer depends only on the shape. The
    cache is keyed by the numbers that define the wing, so a design change misses it and the
    polar is re-solved -- which is the behaviour that keeps this exact rather than a surrogate.
    """
    from openavl import Aircraft, AVLSolver

    planform = TrapezoidalPlanform(area=area, aspect_ratio=aspect_ratio, sweep=sweep, taper=taper)
    aircraft = _lattice(planform, mach, chordwise, spanwise, Aircraft)

    samples = (0.2, 0.5, 0.8)
    drags = []
    for lift in samples:
        solver = AVLSolver(aircraft, cd0=0.0, rho=1.225)
        solver.set_parameter("cl", lift)
        solver.setup_trim(mode=1)
        solver.execute_run(max_iter=30)
        results = solver.get_results()
        # CDFF, not CD. openavl reports both: the near-field pressure sum and the Trefftz-plane
        # far-field value. Near-field induced drag is unreliable on a swept wing -- it is why
        # this model first reported a span efficiency of 1.04 for the shipped planform, which is
        # not physical for a planar wing. The far-field value gives 0.99, and openavl's own
        # SPANEF agrees. AVL's own documentation prefers the far-field drag for the same reason.
        drags.append(float(results["CDFF"]))

    return quadratic_polar(samples, drags, describes=f"a wing of area {area:.4g} m2, AR {aspect_ratio:.4g}")


@lru_cache(maxsize=32)
def _lattice_span_efficiency(
    area: float,
    aspect_ratio: float,
    sweep: float,
    taper: float,
    mach: float,
    chordwise: int,
    spanwise: int,
) -> float:
    """Return the span efficiency openavl reports for this wing, at a mid-range lift.

    Read from openavl's ``SPANEF`` rather than derived from the fitted polar. It is the
    dependency's own number, computed in its Trefftz plane, and re-deriving it here would be
    reimplementing a formula that already exists a layer down -- with the chance of disagreeing
    with it.
    """
    from openavl import Aircraft, AVLSolver

    planform = TrapezoidalPlanform(area=area, aspect_ratio=aspect_ratio, sweep=sweep, taper=taper)
    aircraft = _lattice(planform, mach, chordwise, spanwise, Aircraft)

    solver = AVLSolver(aircraft, cd0=0.0, rho=1.225)
    solver.set_parameter("cl", 0.5)
    solver.setup_trim(mode=1)
    solver.execute_run(max_iter=30)
    return float(solver.get_results()["SPANEF"])


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
    mach : float, optional
        Mach number the lattice is solved at, through its Prandtl-Glauert correction. Default 0.
    chordwise, spanwise : int, optional
        Lattice density. Defaults are deliberately modest; the polar is fitted once per geometry,
        so the cost is bounded whatever the mission length.

    Raises
    ------
    LoadsError
        If openavl is not installed, or if the planform is not one this model can build a lattice
        from, or if the lattice returns a polar that curves the wrong way.
    """

    model_name: ClassVar[str] = "openavl_vortex_lattice"

    __slots__ = ("_mach", "_planform", "_resolution", "_zero_lift_drag")

    def __init__(
        self,
        planform: Planform,
        zero_lift_drag: object = 0.0,
        mach: float = 0.0,
        chordwise: int = 6,
        spanwise: int = 20,
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
        self._mach = float(mach)
        self._resolution = (int(chordwise), int(spanwise))

    @classmethod
    def build(cls, *, planform, span_efficiency, zero_lift_drag):
        """Build the lattice on this wing. ``span_efficiency`` is ignored, and that is the point.

        A case file flying the parabolic polar must state ``ac|aero|polar|e`` as an assumption.
        Here the lattice computes it, so the stated value is deliberately not consulted -- see
        :attr:`span_efficiency` for what the wing actually achieves.
        """
        return cls(planform=planform, zero_lift_drag=zero_lift_drag)

    # -- what the lattice says about this wing -------------------------------------------

    def _polar(self) -> tuple[float, float, float]:
        """Return the fitted ``(cd_min, curvature, cl_at_min_drag)`` for this wing."""
        return _fit_polar(
            self._planform.area,
            self._planform.aspect_ratio,
            self._planform.sweep,
            self._planform.taper,
            self._mach,
            *self._resolution,
        )

    @property
    def span_efficiency(self) -> float:
        """The Oswald efficiency the lattice implies, from the fitted curvature.

        Reported rather than assumed. A case file flying the parabolic polar has to state a value
        for ``ac|aero|polar|e``; this is what the wing actually achieves, and comparing the two is
        the point of installing a lattice at all.
        """
        return _lattice_span_efficiency(
            self._planform.area,
            self._planform.aspect_ratio,
            self._planform.sweep,
            self._planform.taper,
            self._mach,
            *self._resolution,
        )

    def coefficients(self, condition: FlightCondition, planform: Planform) -> AeroCoefficients:
        """Return the coefficients at every point, from the polar fitted to this wing.

        ``planform`` is accepted for the interface's sake and checked against the one the lattice
        was built on, because a silent mismatch would report the drag of a different aeroplane.
        """
        if planform is not self._planform and (
            planform.area != self._planform.area or planform.aspect_ratio != self._planform.aspect_ratio
        ):
            raise LoadsError(
                "This model was built on a different wing than it is being evaluated on: the "
                "lattice describes one geometry and cannot be reused for another."
            )
        cd_min, curvature, cl_at_min_drag = self._polar()
        lift = condition.CL
        drag = self._zero_lift_drag + cd_min + curvature * (lift - cl_at_min_drag) ** 2
        return AeroCoefficients(CL=lift, CD=drag)

    def drag_gradients(self, condition: FlightCondition, planform: Planform) -> dict[str, np.ndarray]:
        """Return the derivatives of ``CD``.

        ``CL`` and ``CD0`` are exact: the fitted polar is a quadratic, and its lift derivative is
        the derivative of that quadratic.

        ``e`` is **zero, and that is correct** -- not missing. This model does not read
        ``ac|aero|polar|e``; it computes the span efficiency from the lattice. A case file's stated
        value has no influence on this drag, so the derivative with respect to it genuinely is
        nothing. Compare :meth:`~cdadt.models.polar.PolarLoads.drag_gradients`, where it is the
        dominant term.

        ``AR`` is an **approximation, and its size is measured**. Writing the fitted curvature as
        :math:`k = 1/(\\pi e A\\!R)` gives :math:`\\partial C_D/\\partial A\\!R = -k(C_L-C_{L_0})^2/A\\!R`
        with the lattice's span efficiency held fixed. The exact derivative carries a second term
        in :math:`\\partial e/\\partial A\\!R`, which for the shipped planform is
        :math:`-1.85\\times10^{-3}` -- 1.8% of the term retained here. So this captures 98% of the
        geometry sensitivity, and ``test_the_neglected_span_efficiency_gradient_stays_small``
        measures that rather than trusting it.

        Making it exact means chaining openavl's ``jacrev`` over ``GeometryDesignParams`` with the
        analytic section derivatives of :class:`~cdadt.models.planform.TrapezoidalPlanform`, and
        letting the reference area move with the wing -- which ``snapshot_refs`` currently holds
        fixed. That is the next piece of work, not a line of it.
        """
        _, curvature, cl_at_min_drag = self._polar()
        lift = condition.CL
        excess = lift - cl_at_min_drag
        return {
            "CL": 2.0 * curvature * excess,
            "CD0": np.ones_like(lift),
            "e": np.zeros_like(lift),
            "AR": -curvature * excess**2 / planform.aspect_ratio,
        }
