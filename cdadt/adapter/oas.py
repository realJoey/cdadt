"""Aerodynamic loads from OpenAeroStruct: the second vortex lattice behind the same slot.

cdadt's aerodynamics is a *slot*, not a solver. :class:`~cdadt.adapter.avl.OpenAVLLoads` fills it
with openavl; this module fills it with OpenAeroStruct, reached the only way cdadt is allowed to
reach it -- through OpenConcept's own ``VLM`` and ``TrapezoidalPlanformMesh``, which live in
``openconcept.aerodynamics.openaerostruct``. **cdadt never imports OpenAeroStruct itself**, and a
contract test with no adapter exemption enforces that.

Why a second one is worth having
--------------------------------

Not redundancy. Two independent codes behind one interface is what turns "the lattice says 0.99"
into a claim that can be checked: on the shipped wing the two agree on induced drag to **0.67%**,
which is the strongest evidence cdadt has that its aerodynamics is driven correctly. A study can
now make that comparison itself by changing one line of a case file, rather than reading it here.

The two are not interchangeable, and the differences are the interesting part:

====================  =========================  ==========================
Property              openavl                    OpenAeroStruct
====================  =========================  ==========================
Induced drag          Trefftz-plane far field    near-field panel sum
Span efficiency       reported (``SPANEF``)      inferred from the fit
Compressibility       Prandtl-Glauert            none; ``CDi`` is Mach-free
Derivatives           ``jax.jacrev``             OpenMDAO analytic totals
Polar in lift         an identity, exact 1e-12   a fit, good to **2e-3**
====================  =========================  ==========================

Two rows deserve more than a table cell.

**Near field.** Near-field induced drag is under-predicted on a swept wing badly enough that *both*
codes imply a span efficiency above 1 for this planar wing, which is impossible since elliptical
loading is optimal. openavl offers a far-field value and cdadt uses it; OpenAeroStruct's ``VLM``
publishes only ``fltcond|CDi``, so this model reads 1.034 and is optimistic on induced drag by about
4% against the openavl model. It says so rather than correcting it with a fudge factor.

**The polar is a fit here, not an identity.** For openavl the quadratic is exact because the
Trefftz-plane drag is a quadratic form in a circulation that is affine in angle of attack -- three
solves determine it and the residual is round-off. The near-field sum carries no such guarantee, and
it shows: measured at three angles the fit never sampled, the worst error is **2.2e-3 relative**,
around :math:`C_L` 0.16 where the drag is smallest. That is small enough to be usable and large
enough that it must be published rather than assumed, which is what
``test_the_openaerostruct_polar_reports_its_own_fit_error`` does.

How the derivatives are exact
-----------------------------

OpenMDAO already has them. ``TrapezoidalPlanformMesh`` and ``VLM`` both declare analytic partials,
so ``compute_totals`` gives :math:`\\partial C_L/\\partial g` and :math:`\\partial C_{D_i}/\\partial g`
for the four wing numbers directly. Those are differentiated through the polar fit here -- the fit
is a linear solve, so its derivative is another linear solve -- giving the geometry Jacobian of the
*fitted polar*, which is what every mission node is evaluated from.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from cdadt.adapter.lattice import GEOMETRY_VARIABLES, POLAR_COEFFICIENTS, LatticePolar, LatticeSolver
from cdadt.models.coefficients import AeroCoefficients
from cdadt.models.loads import AerodynamicLoads, FlightCondition, LoadsError, Planform
from cdadt.models.planform import TrapezoidalPlanform

__all__ = ["ANGLES_OF_ATTACK_DEG", "OpenAeroStructLattice", "OpenAeroStructLoads", "openaerostruct_is_available"]

#: Angles of attack, in **degrees**, the polar is determined from. Degrees rather than the radians
#: openavl works in because that is the unit OpenConcept's ``VLM`` declares ``fltcond|alpha`` in, and
#: converting here would be a units bug waiting to happen. Three is what a quadratic needs.
ANGLES_OF_ATTACK_DEG: tuple[float, ...] = (0.0, 4.5, 9.0)

#: Mach and altitude the lattice is solved at. Neither changes the answer -- OpenAeroStruct's induced
#: drag is Mach-free, verified by solving at two Mach numbers and getting identical ``CDi`` -- but
#: both must be non-zero: the code works in true airspeed and derives it from Mach and altitude, so
#: zero Mach is a zero velocity and every coefficient comes back NaN.
SOLVE_MACH: float = 0.3
SOLVE_ALTITUDE: float = 0.0


def openaerostruct_is_available() -> bool:
    """Return whether OpenConcept's OpenAeroStruct-backed components can be imported.

    Reached through OpenConcept rather than by importing OpenAeroStruct, which cdadt may not do. The
    subpackage raises on import when OpenAeroStruct is absent, so this is the honest probe.
    """
    try:
        from openconcept.aerodynamics.openaerostruct import VLM  # noqa: F401
    except ImportError:
        return False
    return True


class OpenAeroStructLattice(LatticeSolver):
    """OpenConcept's OpenAeroStruct vortex lattice, solved for one wing and fitted to a polar.

    Parameters
    ----------
    planform : TrapezoidalPlanform
        The wing. Its four numbers drive ``TrapezoidalPlanformMesh``, which is parameterised by
        exactly the same four -- which is why the two codes can be put on an identical wing.
    mach : float
        Accepted for the interface and **not used**: this lattice's induced drag is Mach-free. It is
        recorded so the polar can say what it was asked for.
    chordwise, spanwise : int
        Mesh panels, streamwise and on the half span.

    Raises
    ------
    LoadsError
        If OpenAeroStruct is not installed.
    """

    __slots__ = ("_mach", "_planform", "_problem", "_thickness")

    #: The mesh inputs, in the order the Jacobian's columns follow. These are OpenConcept's names for
    #: the same four numbers :data:`~cdadt.adapter.lattice.GEOMETRY_VARIABLES` names.
    MESH_INPUTS: ClassVar[tuple[str, ...]] = ("mesh.S", "mesh.AR", "mesh.sweep", "mesh.taper")

    def __init__(self, planform: TrapezoidalPlanform, mach: float, chordwise: int, spanwise: int) -> None:
        if not openaerostruct_is_available():
            raise LoadsError(
                "OpenAeroStruct is not installed, so OpenConcept's vortex lattice cannot be used. "
                'Install it with: pip install -e ".[transonic]"'
            )
        if not isinstance(planform, TrapezoidalPlanform):
            raise LoadsError(
                f"The mesh is built from a trapezoidal planform; got {type(planform).__name__}, "
                f"which does not describe one."
            )
        self._planform = planform
        self._mach = float(mach)
        self._thickness = 0.1
        self._problem = self._build(chordwise, spanwise)

    def _build(self, chordwise: int, spanwise: int) -> Any:
        """Assemble the mesh generator and the lattice into one problem, ready to differentiate."""
        import openmdao.api as om
        from openconcept.aerodynamics.openaerostruct import VLM, TrapezoidalPlanformMesh

        problem = om.Problem(reports=False)
        problem.model.add_subsystem(
            "mesh", TrapezoidalPlanformMesh(num_x=chordwise, num_y=spanwise), promotes_outputs=[("mesh", "OAS_mesh")]
        )
        problem.model.add_subsystem(
            "vlm",
            # Induced drag alone. OpenAeroStruct's viscous and wave terms are real models, but they
            # are not this one's business: the parasite drag is OpenConcept's component buildup and
            # the wave drag is its Korn model, both added outside. Leaving them on here would count
            # each of them twice.
            VLM(num_x=chordwise, num_y=spanwise, surf_options={"with_viscous": False, "with_wave": False}),
            promotes_inputs=[("ac|geom|wing|OAS_mesh", "OAS_mesh")],
        )
        problem.setup(mode="rev")

        problem.set_val("mesh.S", self._planform.area, units="m**2")
        problem.set_val("mesh.AR", self._planform.aspect_ratio)
        problem.set_val("mesh.sweep", self._planform.sweep, units="deg")
        problem.set_val("mesh.taper", self._planform.taper)
        problem.set_val("vlm.ac|geom|wing|toverc", np.full(spanwise, self._thickness))
        problem.set_val("vlm.fltcond|M", SOLVE_MACH)
        problem.set_val("vlm.fltcond|h", SOLVE_ALTITUDE, units="m")
        return problem

    # -- solving and differentiating ------------------------------------------------------

    def coefficients_at(self, angle_of_attack_deg: float) -> tuple[float, float]:
        """Return ``(CL, CDi)`` at one angle of attack, in degrees."""
        self._problem.set_val("vlm.fltcond|alpha", angle_of_attack_deg, units="deg")
        self._problem.run_model()
        return float(self._problem.get_val("vlm.fltcond|CL")[0]), float(self._problem.get_val("vlm.fltcond|CDi")[0])

    def _totals_at(self, angle_of_attack_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dCL/dg, dCDi/dg)`` at one angle of attack, from OpenMDAO's own derivatives.

        Analytic, not differenced: both components declare their partials, so this is the chain rule
        through the mesh generator and the lattice, computed by the framework that owns them.
        """
        self._problem.set_val("vlm.fltcond|alpha", angle_of_attack_deg, units="deg")
        self._problem.run_model()
        totals = self._problem.compute_totals(of=["vlm.fltcond|CL", "vlm.fltcond|CDi"], wrt=list(self.MESH_INPUTS))
        lift = np.array([float(totals[("vlm.fltcond|CL", name)][0, 0]) for name in self.MESH_INPUTS])
        drag = np.array([float(totals[("vlm.fltcond|CDi", name)][0, 0]) for name in self.MESH_INPUTS])
        return lift, drag

    def fit(self) -> LatticePolar:
        """Solve at three angles, fit the polar, and differentiate the fit with respect to the wing.

        The fit is a determined linear system in :math:`[a, b, c]` for
        :math:`C_D = a C_L^2 + b C_L + c`, so differentiating it is another solve with the same
        matrix:

        .. math::

           V \\frac{\\partial \\mathbf{x}}{\\partial g}
             = \\frac{\\partial \\mathbf{C_D}}{\\partial g}
             - \\frac{\\partial V}{\\partial g}\\mathbf{x}

        where :math:`\\partial V/\\partial g` is non-zero only because the lift at a fixed angle of
        attack moves when the wing does. Dropping that term is the mistake that would make these
        derivatives silently approximate.
        """
        samples = [self.coefficients_at(angle) for angle in ANGLES_OF_ATTACK_DEG]
        lift = np.array([point[0] for point in samples])
        drag = np.array([point[1] for point in samples])

        vandermonde = np.stack([lift**2, lift, np.ones_like(lift)], axis=1)
        quadratic, linear, constant = np.linalg.solve(vandermonde, drag)
        lift_at_minimum = -linear / (2.0 * quadratic)
        minimum_drag = constant - quadratic * lift_at_minimum**2

        totals = [self._totals_at(angle) for angle in ANGLES_OF_ATTACK_DEG]
        lift_gradients = np.stack([point[0] for point in totals])
        drag_gradients = np.stack([point[1] for point in totals])

        jacobian = np.zeros((len(POLAR_COEFFICIENTS), len(GEOMETRY_VARIABLES)))
        for column in range(len(GEOMETRY_VARIABLES)):
            moved = np.stack(
                [2.0 * lift * lift_gradients[:, column], lift_gradients[:, column], np.zeros_like(lift)], axis=1
            )
            d_quadratic, d_linear, d_constant = np.linalg.solve(
                vandermonde, drag_gradients[:, column] - moved @ np.array([quadratic, linear, constant])
            )
            d_lift_at_minimum = -d_linear / (2.0 * quadratic) + linear * d_quadratic / (2.0 * quadratic**2)
            jacobian[0, column] = (
                d_constant - d_quadratic * lift_at_minimum**2 - 2.0 * quadratic * lift_at_minimum * d_lift_at_minimum
            )
            jacobian[1, column] = d_quadratic
            jacobian[2, column] = d_lift_at_minimum

        return LatticePolar(
            minimum_drag=float(minimum_drag),
            curvature=float(quadratic),
            lift_at_minimum_drag=float(lift_at_minimum),
            # Inferred, not reported: OpenAeroStruct's VLM publishes no span efficiency, so this is
            # the definition rather than the code's own number. That is the substantive difference
            # from openavl, where the identity k = 1/(pi e AR) is a *check* because e is measured
            # independently. Here it can only be a conversion.
            span_efficiency=float(1.0 / (np.pi * quadratic * self._planform.aspect_ratio)),
            jacobian=tuple(tuple(float(value) for value in row) for row in jacobian),
            describes=(
                f"an OpenAeroStruct mesh of area {self._planform.area:.4g} m2, " f"AR {self._planform.aspect_ratio:.4g}"
            ),
        )


class OpenAeroStructLoads(AerodynamicLoads):
    """Lift-dependent drag from OpenConcept's OpenAeroStruct vortex lattice.

    The same shape as :class:`~cdadt.adapter.avl.OpenAVLLoads` and deliberately so: both fit a polar
    per geometry, both evaluate every mission node in closed form, both report exact geometry
    derivatives, and both are chosen by one line of a case file. What differs is the code underneath
    and what it can see -- see this module's own documentation, and :doc:`/truth` for the comparison
    between them.

    Parameters
    ----------
    planform : Planform
        The wing to mesh. Must be a :class:`~cdadt.models.planform.TrapezoidalPlanform`.
    zero_lift_drag : array_like, optional
        Parasite ``CD0`` from outside the lattice. Default 0.
    chordwise, spanwise : int, optional
        Mesh density.
    library : LatticeLibrary, optional
        Where solved polars are kept. Injected by the analysis group so that all fourteen mission
        phases share one; a private one is made when none is offered.

    Raises
    ------
    LoadsError
        If OpenAeroStruct is absent, or the planform is not one a mesh can be built from.
    """

    model_name: ClassVar[str] = "openaerostruct_vortex_lattice"

    #: The mesh is rebuilt from all four wing numbers. ``e`` is absent for the same reason as in the
    #: openavl model: this one derives the span efficiency rather than reading a case file's.
    DRAG_DEPENDS_ON: ClassVar[tuple[str, ...]] = ("CL", "CD0", "area", "AR", "sweep", "taper")

    __slots__ = ("_library", "_planform", "_resolution", "_zero_lift_drag")

    def __init__(
        self,
        planform: Planform,
        zero_lift_drag: object = 0.0,
        chordwise: int = 6,
        spanwise: int = 20,
        library: Any = None,
    ) -> None:
        from cdadt.adapter.lattice import LatticeLibrary

        if not isinstance(planform, TrapezoidalPlanform):
            raise LoadsError(
                f"The mesh is built from a trapezoidal planform; got {type(planform).__name__}, "
                f"which does not describe one."
            )
        self._planform = planform
        self._zero_lift_drag = np.atleast_1d(np.asarray(zero_lift_drag, dtype=float))
        self._resolution = (int(chordwise), int(spanwise))
        self._library = library if library is not None else LatticeLibrary(solver=OpenAeroStructLattice)

    @classmethod
    def build(
        cls,
        *,
        planform: Planform,
        span_efficiency: float,
        zero_lift_drag: object,
        workspace: object = None,
    ) -> OpenAeroStructLoads:
        """Build the mesh on this wing. ``span_efficiency`` is ignored: the lattice implies one.

        ``workspace`` must be a :class:`~cdadt.adapter.lattice.LatticeLibrary` **solving with this
        model's own solver**. A library is a store of solved polars, and a polar solved by openavl is
        not interchangeable with one solved by OpenAeroStruct -- they differ by 5% on this wing,
        because one is a far-field value and the other near-field. Handing this model the other
        code's library would silently report the other code's drag, so it is refused by name.
        """
        from cdadt.adapter.lattice import LatticeLibrary

        if workspace is not None:
            if not isinstance(workspace, LatticeLibrary):
                raise LoadsError(
                    f"This model keeps its solved polars in a LatticeLibrary; got "
                    f"{type(workspace).__name__}, which it cannot store anything in."
                )
            if workspace.solver is not OpenAeroStructLattice:
                raise LoadsError(
                    f"This model was offered a library solving with {workspace.solver.__name__}, "
                    f"whose polars are a different code's answer for the same wing. A library "
                    f"belongs to one solver."
                )
        return cls(planform=planform, zero_lift_drag=zero_lift_drag, library=workspace)

    # -- what the lattice says about this wing -------------------------------------------

    def polar(self) -> LatticePolar:
        """Return the fitted, differentiated polar for this wing.

        No Mach argument, and no interpolation across Mach either -- unlike the openavl model, which
        needs both. This lattice's induced drag does not depend on Mach at all, so a Mach sweep would
        be fifteen solves producing five identical answers.
        """
        return self._library.polar_for(self._planform, SOLVE_MACH, *self._resolution)

    @property
    def span_efficiency(self) -> float:
        """The Oswald efficiency this lattice's polar implies."""
        return self.polar().span_efficiency

    def _check_planform(self, planform: Planform) -> None:
        """Reject a planform the mesh was not built on, which would report a different aeroplane."""
        if planform is not self._planform and (
            planform.area != self._planform.area or planform.aspect_ratio != self._planform.aspect_ratio
        ):
            raise LoadsError(
                "This model was built on a different wing than it is being evaluated on: the "
                "mesh describes one geometry and cannot be reused for another."
            )

    def coefficients(self, condition: FlightCondition, planform: Planform) -> AeroCoefficients:
        """Return the coefficients at every point, from the polar fitted to this wing."""
        self._check_planform(planform)
        polar = self.polar()
        lift = condition.CL
        return AeroCoefficients(CL=lift, CD=self._zero_lift_drag + polar.drag_at(lift))

    def drag_gradients(self, condition: FlightCondition, planform: Planform) -> dict[str, np.ndarray]:
        """Return the exact derivatives of ``CD``, from OpenMDAO's totals through the polar fit.

        The same chain rule the openavl model applies, over a Jacobian obtained a different way:

        .. math::

           \\frac{\\partial C_D}{\\partial g} =
               \\frac{\\partial C_{D_\\mathrm{min}}}{\\partial g}
             + \\frac{\\partial k}{\\partial g}\\,(C_L - C_{L_\\mathrm{minD}})^2
             - 2k\\,(C_L - C_{L_\\mathrm{minD}})\\,\\frac{\\partial C_{L_\\mathrm{minD}}}{\\partial g}
        """
        self._check_planform(planform)
        polar = self.polar()
        lift = condition.CL
        excess = lift - polar.lift_at_minimum_drag

        gradients = {
            "CL": 2.0 * polar.curvature * excess,
            "CD0": np.ones_like(lift),
            "e": np.zeros_like(lift),
        }
        for variable in LatticePolar.GEOMETRY:
            d_minimum, d_curvature, d_lift_at_minimum = polar.gradient(variable)
            gradients[variable] = (
                np.full_like(lift, d_minimum)
                + d_curvature * excess**2
                - 2.0 * polar.curvature * excess * d_lift_at_minimum
            )
        return gradients
