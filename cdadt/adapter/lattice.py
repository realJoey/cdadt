"""openavl's differentiable vortex lattice, driven from a cdadt planform.

This module exists so that :mod:`cdadt.adapter.avl` can state a drag polar *and its exact
derivatives with respect to the wing* from one source. It owns every openavl call cdadt makes;
nothing here is copied from openavl and nothing is patched.

Why the far field, and why the fit is exact
-------------------------------------------

openavl reports induced drag twice: a near-field sum of pressures on the panels (``CD``) and a
Trefftz-plane far-field value (``CDFF``). They are not interchangeable. Read together with the
matching lift -- ``CLFF`` with ``CDFF`` -- the far-field pair satisfies

.. math::

   C_{D_\\mathrm{FF}} = \\frac{C_{L_\\mathrm{FF}}^2}{\\pi\\,e\\,A\\!R}

with openavl's own reported span efficiency, to machine precision. That identity is what
``test_the_fitted_curvature_is_openavls_own_span_efficiency`` checks, and it is the reason this
model is fitted on the far-field pair: any other pairing produces a polar the dependency itself
disagrees with. cdadt got this wrong twice -- first by taking near-field ``CD``, which claimed a
span efficiency of 1.056 for a planar wing, then by pairing far-field ``CDFF`` with the *commanded*
near-field lift, which biased the curvature by 2.2% and left the model's own reported ``e`` of
0.990 inconsistent with the 0.969 its drag implied.

A vortex lattice is linear: the circulation is affine in angle of attack, the far-field lift is
linear in it and the far-field drag is a quadratic form in the circulation. So

.. math::

   C_D = C_{D_\\mathrm{min}} + k\\,(C_L - C_{L_\\mathrm{minD}})^2

is not a curve fit but an identity, and three solves determine it exactly. The residual of a fourth
sample measures that claim rather than assuming it
(``test_the_lattice_polar_is_exactly_quadratic``).

Why the derivatives come from openavl and not from a formula
------------------------------------------------------------

Writing :math:`k = 1/(\\pi e A\\!R)` and holding :math:`e` fixed gives an aspect-ratio derivative
that is wrong by 1.8% for the shipped wing, and gives *nothing at all* for sweep and taper -- which
an optimizer reads as "these do not matter". Both are avoided here by differentiating the lattice
itself: :func:`openavl.jax.geom_jax.update_geometry` rebuilds the panels inside JAX from the four
numbers a case file declares, so ``jax.jacrev`` returns the derivatives of the fitted polar with
respect to area, aspect ratio, sweep and taper directly.

Reverse mode, not forward: openavl's circulation solve is registered as a ``custom_vjp``, so
``jax.jacfwd`` raises *"can't apply forward-mode autodiff (jvp) to a custom_vjp function"*. Three
reverse passes cost the same as four forward ones here anyway.

What is held fixed
------------------

Twist and dihedral. The sections' incidence angles are taken from the baseline lattice and are not
design variables, because no case file declares them; the reference area, span and chord *do* move
with the wing, which they must, or the coefficients would be normalised by a different aeroplane
than the one being solved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from cdadt.models.loads import LoadsError
from cdadt.models.planform import TrapezoidalPlanform

__all__ = ["ANGLES_OF_ATTACK", "DifferentiableLattice", "LatticeLibrary", "LatticePolar", "LatticeSolver"]

#: Angles of attack, in radians, the polar is determined from. Three is the exact number of samples
#: a quadratic needs; they are spread across the lift range a transport mission uses (roughly
#: :math:`C_L` 0 to 0.75 for the shipped wing) so that the conditioning of the solve is good, not
#: because the answer depends on where they sit.
ANGLES_OF_ATTACK: tuple[float, ...] = (0.0, 0.08, 0.16)

#: The wing numbers the polar is differentiated with respect to, in the order the Jacobian's
#: columns follow. These are exactly the four a case file declares for a trapezoidal wing.
GEOMETRY_VARIABLES: tuple[str, ...] = ("area", "AR", "sweep", "taper")

#: The polar's three coefficients, in the order the Jacobian's rows follow.
POLAR_COEFFICIENTS: tuple[str, ...] = ("minimum_drag", "curvature", "lift_at_minimum_drag")


class LatticePolar:
    """The drag polar of one wing at one Mach number, with its geometry Jacobian.

    Read-only because it is cached and shared: every node of every mission phase within one design
    iteration reads the same object, and a mutable one would let a caller change what the next
    reader sees.

    Parameters
    ----------
    minimum_drag : float
        :math:`C_{D_\\mathrm{min}}`, the lowest lift-dependent drag the wing achieves. Zero for an
        untwisted, uncambered wing, and openavl says so exactly rather than approximately.
    curvature : float
        :math:`k`, so that the drag grows as :math:`k` times the squared lift excess.
    lift_at_minimum_drag : float
        :math:`C_{L_\\mathrm{minD}}`, where the polar bottoms out. Non-zero only with twist or
        camber.
    span_efficiency : float
        openavl's own ``SPANEF``, read rather than re-derived. Related to the curvature by
        :math:`k = 1/(\\pi e A\\!R)`.
    jacobian : tuple of tuple of float
        Derivatives of the three coefficients, in :data:`POLAR_COEFFICIENTS` order, with respect to
        the four wing numbers, in :data:`GEOMETRY_VARIABLES` order.
    describes : str, optional
        What wing this is the polar of, for error messages.

    Raises
    ------
    LoadsError
        If the curvature is not positive. Validated on construction rather than at the call site so
        that a polar which curves the wrong way cannot exist: induced drag grows with lift, and a
        negative curvature reaching an optimizer is worse than an error -- it becomes an incentive
        to add lift for less drag, and the design walks off into a region the solver invented.
    """

    COEFFICIENTS: ClassVar[tuple[str, ...]] = POLAR_COEFFICIENTS
    GEOMETRY: ClassVar[tuple[str, ...]] = GEOMETRY_VARIABLES

    __slots__ = ("_describes", "_jacobian", "_values")

    def __init__(
        self,
        minimum_drag: float,
        curvature: float,
        lift_at_minimum_drag: float,
        span_efficiency: float,
        jacobian: tuple[tuple[float, ...], ...],
        describes: str = "this wing",
    ) -> None:
        if curvature <= 0.0:
            raise LoadsError(
                f"The lattice produced a drag polar with non-positive curvature ({curvature:.3e}) "
                f"for {describes}. Induced drag must grow with lift, so this is a geometry the "
                f"lattice could not resolve rather than a result."
            )
        self._values = (float(minimum_drag), float(curvature), float(lift_at_minimum_drag), float(span_efficiency))
        self._jacobian = tuple(tuple(float(value) for value in row) for row in jacobian)
        self._describes = str(describes)

    @property
    def minimum_drag(self) -> float:
        """:math:`C_{D_\\mathrm{min}}`, the lowest lift-dependent drag this wing achieves."""
        return self._values[0]

    @property
    def curvature(self) -> float:
        """:math:`k`: the drag grows as :math:`k` times the squared lift excess."""
        return self._values[1]

    @property
    def lift_at_minimum_drag(self) -> float:
        """:math:`C_{L_\\mathrm{minD}}`, the lift at which the polar bottoms out."""
        return self._values[2]

    @property
    def span_efficiency(self) -> float:
        """openavl's own ``SPANEF`` for this wing, read rather than re-derived."""
        return self._values[3]

    @property
    def jacobian(self) -> tuple[tuple[float, ...], ...]:
        """Derivatives of the three coefficients with respect to the four wing numbers."""
        return self._jacobian

    @property
    def describes(self) -> str:
        """What wing this is the polar of."""
        return self._describes

    @property
    def coefficients(self) -> tuple[float, float, float]:
        """The three polar coefficients, in :data:`POLAR_COEFFICIENTS` order."""
        return self._values[:3]

    def __repr__(self) -> str:
        """Return a representation naming the wing and the curvature."""
        return f"LatticePolar({self._describes!r}, curvature={self.curvature:.6g}, e={self.span_efficiency:.6g})"

    def gradient(self, variable: str) -> tuple[float, float, float]:
        """Return how the three coefficients change with one wing number.

        Parameters
        ----------
        variable : str
            One of :data:`GEOMETRY_VARIABLES`.

        Returns
        -------
        tuple of float
            ``(d minimum_drag, d curvature, d lift_at_minimum_drag)`` per unit of ``variable``.

        Raises
        ------
        KeyError
            If ``variable`` is not one this polar was differentiated with respect to. Silently
            returning zero would be indistinguishable from a wing number that genuinely does not
            matter.
        """
        if variable not in self.GEOMETRY:
            raise KeyError(
                f"The lattice polar carries no derivative with respect to {variable!r}; it was "
                f"differentiated with respect to {', '.join(self.GEOMETRY)}."
            )
        column = self.GEOMETRY.index(variable)
        return tuple(float(row[column]) for row in self.jacobian)  # type: ignore[return-value]

    def drag_at(self, lift: np.ndarray) -> np.ndarray:
        """Evaluate the polar at one or many lift coefficients."""
        return self.minimum_drag + self.curvature * (lift - self.lift_at_minimum_drag) ** 2


class LatticeSolver(ABC):
    """A vortex-lattice code, reduced to the one thing cdadt asks of it.

    Two exist -- openavl through :class:`DifferentiableLattice`, and OpenAeroStruct through
    :class:`~cdadt.adapter.oas.OpenAeroStructLattice` -- and they share nothing but this interface
    and the value object it returns. That is the point of it: a study picks its aerodynamics by
    naming a class, and the machinery around it does not know which code is underneath.

    A solver is constructed for **one wing at one Mach number and one lattice density**, because
    both codes snapshot geometry at construction. :meth:`fit` is what costs, and
    :class:`LatticeLibrary` is what makes sure it is paid once per wing per study.
    """

    @abstractmethod
    def fit(self) -> LatticePolar:
        """Solve the lattice and return its polar, with the exact geometry Jacobian.

        Returns
        -------
        LatticePolar
            Which refuses its own construction if the curvature is not positive, so an
            implementation does not have to check that itself.
        """


class DifferentiableLattice(LatticeSolver):
    """A vortex lattice on a trapezoidal wing, differentiable with respect to that wing.

    One instance describes one wing at one Mach number and one lattice density. It holds the
    snapshot openavl needs to rebuild geometry inside JAX -- the panel topology and the baseline
    arrays -- which is why it is an object rather than a function: that snapshot costs a lattice
    solve, and the primal fit and its Jacobian must be taken from the same one or they would
    describe subtly different lattices.

    Parameters
    ----------
    planform : TrapezoidalPlanform
        The wing. Its four numbers become the JAX inputs the polar is differentiated against.
    mach : float
        Freestream Mach number, reaching the lattice through openavl's Prandtl-Glauert correction.
    chordwise, spanwise : int
        Panel counts on the half-wing.

    Raises
    ------
    LoadsError
        If openavl is not installed.
    """

    __slots__ = ("_baseline", "_controls", "_incidences", "_mach", "_planform", "_symmetry", "_topology")

    def __init__(self, planform: TrapezoidalPlanform, mach: float, chordwise: int, spanwise: int) -> None:
        try:
            from openavl import Aircraft, AVLSolver
            from openavl.jax import (
                design_params_from_state,
                snapshot_analysis_geometry,
                snapshot_refs,
                snapshot_topology,
            )
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise LoadsError(
                "openavl is not installed, so the vortex-lattice loads model cannot be used. "
                'Install it with: pip install -e ".[avl]"'
            ) from error

        self._planform = planform
        self._mach = float(mach)

        solver = AVLSolver(self.aircraft(planform, self._mach, chordwise, spanwise, Aircraft), cd0=0.0, rho=1.225)
        # One iteration is enough: nothing is read from this solve except the panel arrays, and the
        # panels are set by the geometry rather than by the flow.
        solver.execute_run(max_iter=1)

        self._baseline = snapshot_analysis_geometry(solver.state)
        self._topology = snapshot_topology(solver.state, solver.model)
        self._incidences = design_params_from_state(solver.state, solver.model).aincs
        self._symmetry = snapshot_refs(solver.state).iysym
        self._controls = max(1, int(solver.state.ncontrol))

    @staticmethod
    def aircraft(
        planform: TrapezoidalPlanform,
        mach: float,
        chordwise: int,
        spanwise: int,
        aircraft_class: type,
    ) -> object:
        """Build the openavl aircraft for a trapezoidal wing.

        A static method on the class that solves the lattice rather than a free function beside it:
        constructing the lifting surface is an engineering step, and the brief puts those on classes.
        Static because it needs nothing from an instance -- it is what an instance is built *from*,
        so it must run before ``__init__`` has anything to offer.

        ``aircraft_class`` is injected rather than imported here so that this method carries no
        import of its own: the dependency is optional, and only the callers that need it import it.
        The reference quantities are the planform's own, which is what makes the coefficients the
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

    # -- the differentiable chain --------------------------------------------------------

    def _far_field_forces(self, angle_of_attack: Any, design: Any) -> Any:
        """Solve the lattice for one angle of attack and one set of wing numbers, in JAX.

        Composes the same four public openavl calls :func:`run_analysis_with_geometry` composes
        internally. It is not called instead, because it returns an ``AnalysisResult``, which keeps
        only the near-field drag; ``compute_forces`` returns the full ``ForceResult``, with the
        far-field pair and the span efficiency this model is built on.
        """
        from openavl.jax import (
            ReferenceQuantities,
            compute_circulation,
            compute_forces,
            compute_velocities,
            update_geometry,
        )
        from openavl.jax.backend import jnp
        from openavl.jax.types import FlowCondition, GeometryDesignParams, Velocities

        area, aspect_ratio, sweep, taper = design
        # The geometry formulas are cdadt's, stated once, in TrapezoidalPlanform -- evaluated here
        # on JAX tracers by passing jax.numpy in. Re-deriving them here would be the same algebra
        # in two places, free to diverge, and the divergence would show up as a wrong derivative
        # rather than as a wrong shape.
        wing = TrapezoidalPlanform.geometry(area, aspect_ratio, sweep, taper, arrays=jnp)
        mean_chord = wing["mean_aerodynamic_chord"]

        design_params = GeometryDesignParams(
            aincs=self._incidences,
            chords=jnp.stack([wing["root_chord"], wing["tip_chord"]]),
            xles=jnp.stack([jnp.zeros(()), wing["tip_leading_edge"]]),
            yles=jnp.stack([jnp.zeros(()), wing["semi_span"]]),
            zles=jnp.zeros(2),
        )
        references = ReferenceQuantities(
            sref=area,
            cref=mean_chord,
            bref=wing["span"],
            xyzref=jnp.stack([0.25 * mean_chord, jnp.zeros(()), jnp.zeros(())]),
            cdref=jnp.zeros(()),
            iysym=self._symmetry,
        )
        flow = FlowCondition(
            alfa=angle_of_attack,
            beta=jnp.zeros(()),
            wrot=jnp.zeros(3),
            mach=jnp.asarray(self._mach),
            delcon=jnp.zeros(self._controls),
        )

        geometry = update_geometry(self._topology, design_params, self._baseline)
        circulation = compute_circulation(geometry.circulation, flow, references)
        wake, vortex = compute_velocities(geometry.circulation, circulation)
        return compute_forces(
            geometry.force,
            circulation,
            Velocities(vv=vortex, wv=wake),
            flow,
            references,
            geometry.body,
            geometry.trefftz,
        )

    def _polar_coefficients(self, design: Any, angles: tuple[float, ...]) -> tuple[Any, Any]:
        """Return the three polar coefficients, and openavl's span efficiency alongside them.

        The whole reason this is one JAX function rather than a loop in numpy: differentiating it
        gives the derivatives of the *fitted polar*, not of a single solve, so the closed-form
        evaluation every mission node uses is differentiated as written.

        The span efficiency comes back as an auxiliary output rather than a fourth differentiated
        one. It is a diagnostic -- nothing consumes its derivative -- and carrying it in the
        differentiated vector would cost a fourth reverse pass, which is a whole extra lattice
        solve per wing.
        """
        from openavl.jax.backend import jnp

        solved = [self._far_field_forces(jnp.asarray(angle), design) for angle in angles]
        lift = jnp.stack([result.CLFF for result in solved])
        drag = jnp.stack([result.CDFF for result in solved])

        # Three samples, three unknowns in CD = a CL^2 + b CL + c: a determined system, not a
        # least-squares fit, because the lattice polar is exactly quadratic.
        vandermonde = jnp.stack([lift**2, lift, jnp.ones_like(lift)], axis=1)
        quadratic, linear, constant = jnp.linalg.solve(vandermonde, drag)
        lift_at_minimum = -linear / (2.0 * quadratic)
        coefficients = jnp.stack([constant - quadratic * lift_at_minimum**2, quadratic, lift_at_minimum])
        return coefficients, solved[len(solved) // 2].SPANEF

    # -- what callers ask for ------------------------------------------------------------

    def _design_vector(self) -> Any:
        """The four wing numbers, in :data:`GEOMETRY_VARIABLES` order, as a JAX array."""
        from openavl.jax.backend import jnp

        return jnp.asarray(
            [self._planform.area, self._planform.aspect_ratio, self._planform.sweep, self._planform.taper]
        )

    def fit(self) -> LatticePolar:
        """Solve the lattice, fit the polar and differentiate it with respect to the wing.

        One forward pass and three reverse ones, taken from a single :func:`jax.vjp` so that the
        coefficients and their derivatives come from the same solves. Computing the primal
        separately would repeat three lattice solves to arrive at numbers already in hand.

        Returns
        -------
        LatticePolar
            The three coefficients, openavl's span efficiency, and the exact Jacobian. It refuses
            its own construction if the curvature is not positive.
        """
        from openavl.jax.backend import jax

        coefficients, pullback, span_efficiency = jax.vjp(
            lambda values: self._polar_coefficients(values, ANGLES_OF_ATTACK), self._design_vector(), has_aux=True
        )
        values = np.asarray(coefficients, dtype=float)
        # One reverse pass per polar coefficient, each seeded with that coefficient's unit vector.
        jacobian = [np.asarray(pullback(seed)[0], dtype=float) for seed in np.eye(len(POLAR_COEFFICIENTS))]
        return LatticePolar(
            minimum_drag=float(values[0]),
            curvature=float(values[1]),
            lift_at_minimum_drag=float(values[2]),
            span_efficiency=float(span_efficiency),
            jacobian=tuple(tuple(float(value) for value in row) for row in jacobian),
            describes=(
                f"a wing of area {self._planform.area:.4g} m2, AR {self._planform.aspect_ratio:.4g}, "
                f"at M {self._mach:.3g}"
            ),
        )

    def far_field_at(self, angle_of_attack: float) -> tuple[float, float, float]:
        """Return ``(CLFF, CDFF, SPANEF)`` at one angle of attack, for verification.

        Public because the claim that the polar is exactly quadratic is only worth as much as the
        test that measures its residual against extra samples, and that test needs a way in.
        """
        from openavl.jax.backend import jnp

        result = self._far_field_forces(jnp.asarray(angle_of_attack), self._design_vector())
        return float(result.CLFF), float(result.CDFF), float(result.SPANEF)


class LatticeLibrary:
    """The polars solved so far, so that one wing is solved once per study rather than per node.

    This exists because the alternative is module-level state. An ``lru_cache`` on a free function
    is the obvious way to memoise a geometry-keyed solve, and it is what this module used to do --
    but a decorator's cache persists across every ``Problem`` in the process, is mutable by anyone
    who can import the module, and hands two independently constructed models the same object. The
    project's brief says to avoid globals entirely and minimise shared mutable state, and
    :doc:`/architecture` claims the suite enforces it; a cache reachable only through the object
    that owns it is what makes both true.

    **Lifetime is the caller's business, and that is the point.** One library per analysis is
    correct: every phase of a mission flies the same wing, so they must share, and a library per
    *component* would re-solve the lattice fourteen times per design. The analysis group creates one
    and injects it; see :class:`~cdadt.adapter.analysis.SizingMissionAnalysis`.

    Not thread-safe, and does not need to be: OpenMDAO evaluates a model serially within a process,
    and a duplicated solve would cost time rather than correctness.
    """

    __slots__ = ("_solved", "_solver")

    def __init__(self, solver: type[LatticeSolver] = DifferentiableLattice) -> None:
        self._solver = solver
        self._solved: dict[tuple[float, float, float, float, float, int, int], LatticePolar] = {}

    @property
    def solver(self) -> type[LatticeSolver]:
        """The vortex-lattice code this library solves with."""
        return self._solver

    def polar_for(
        self,
        planform: TrapezoidalPlanform,
        mach: float,
        chordwise: int,
        spanwise: int,
    ) -> LatticePolar:
        """Return the fitted, differentiated polar for one wing at one Mach number.

        Keyed on the numbers that define the problem, because an optimizer asks for the same wing at
        every node of every phase within one design iteration and the lattice answer depends only on
        the shape and the Mach number. A design change misses and the lattice is re-solved, which is
        what keeps this exact rather than a surrogate that drifts.
        """
        key = (
            float(planform.area),
            float(planform.aspect_ratio),
            float(planform.sweep),
            float(planform.taper),
            float(mach),
            int(chordwise),
            int(spanwise),
        )
        if key not in self._solved:
            self._solved[key] = self._solver(planform, mach, chordwise, spanwise).fit()
        return self._solved[key]

    def __len__(self) -> int:
        """Return how many wings have been solved."""
        return len(self._solved)

    def clear(self) -> None:
        """Forget every solved polar. For a study that wants to measure the solve cost again."""
        self._solved.clear()

    def __repr__(self) -> str:
        """Return a representation naming how much has been solved."""
        return f"LatticeLibrary({self._solver.__name__}, {len(self._solved)} solved)"
