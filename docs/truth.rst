What is checked against what
============================

OpenConcept and openavl are verified and validated codes. cdadt is not a third physics code beside
them; it drives one and reads the other. So for every number cdadt reports there ought to be a
dependency artefact that is its **truth reference** -- a published value, a reference fixture, a
validated function -- and a test that compares the two.

This page is that map, one row per thing cdadt does. It exists because the gaps are the interesting
part: three of the rows below were blank until the map was drawn, and one of them was blank while
looking green.

.. contents::
   :local:
   :depth: 1

Against OpenConcept
-------------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - What cdadt does
     - OpenConcept's truth reference
     - How it is checked
   * - Drives ``B738SizingMissionAnalysis``
     - ``run_738_sizing_analysis`` in ``examples/B738_sizing.py``
     - Imported and **run live**, same process, same grid. 22 quantities plus the full throttle
       history, at **1e-6**. :mod:`tests.test_validation`
   * - Reports the sizing numbers
     - The literals in ``examples/tests/test_example_aircraft.py::B738SizingTestCase`` -- 35213.767
       lbm block fuel, 40991.188 lbm total fuel, 172711.303 lbm MTOW, at ``num_nodes=5``
     - Compared at OpenConcept's own grid and its own 1e-4 tolerance. Measured **1.3e-6**.
       :mod:`tests.test_validation`
   * - Substitutes its own aircraft model
     - ``B738AircraftModel``, and the ``aircraft_model`` option every mission profile declares
       (``mission/profiles.py:33``)
     - The promoted face of both classes is measured and required to be identical --
       ``{drag, thrust, weight}``. :mod:`tests.test_adapter`
   * - Substitutes its own drag
     - ``examples/B738_VLM_drag.py``, OpenConcept's own precedent for swapping the drag model, and
       the ``VLM`` component underneath it
     - **Numerically anchored.** OpenConcept's own vortex lattice is run on the identical
       trapezoidal wing -- ``TrapezoidalPlanformMesh`` takes the same four numbers cdadt's planform
       does -- and the two codes agree on the induced drag to **0.67%**. See below.
       :mod:`tests.test_adapter`
   * - Transonic drag rise
     - ``WaveDragFromSections``, OpenConcept's Korn-equation model, verified against OpenAeroStruct
       by its own ``test_wave_drag.py``
     - **Installed rather than written.** cdadt supplies the sections from its planform and adds the
       result to the parasite drag; off by default, so the reference example is still reproducible.
       :mod:`tests.test_adapter`
   * - Its own analysis group
     - The reference problem, again
     - The **parity anchor**: cdadt's group, cdadt's aircraft model, cdadt's loads component, flying
       the parabolic polar, reproduces the reference to **1e-9**. This is what makes any later
       difference attributable to physics rather than plumbing. :mod:`tests.test_adapter`
   * - Parasite drag, engine deck, weights, maximum lift, tail sizing
     - The components themselves, used unmodified
     - Nothing to check: cdadt instantiates them and changes nothing, so OpenConcept's own tests are
       the whole of their verification. What cdadt owes here is the contract that it *has* not
       changed them, which :mod:`tests.test_boundary` enforces -- clean working tree, no local
       commits touching loaded modules, no subclassing, no copied source

The distinction between the first two rows is the one worth keeping. Running the example live proves
cdadt drives the box faithfully; it cannot prove the box still computes what OpenConcept says it
should, because a regression in the dependency would move both sides together. Only the published
literals catch that, and until they were checked the suite was green without them.

Against openavl
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - What cdadt does
     - openavl's truth reference
     - How it is checked
   * - Reads far-field forces from ``compute_forces``
     - ``tests/jax_backend/core/test_forces.py::test_compute_forces_trefftz_not_double_symmetrized``,
       marked ``reference`` -- openavl's marker for *"numerically validated against Fortran binaries
       or AVL run-case outputs"* -- which pins ``CLFF``, ``CYFF`` and ``CDFF`` against its NumPy
       ``tpforc``
     - Cited, not re-run: it is openavl's test of openavl. What cdadt owes is that it calls that
       function and reads those fields, which it does
   * - Composes the differentiable chain by hand
     - ``run_analysis_with_geometry``, openavl's own composition of the same four public calls, and
       the one its ``OpenAVLComp`` and ``examples/rectangular_wing_opt.py`` use
     - Both are run on the same geometry and must agree on the two quantities they share, lift and
       near-field drag. Measured **3e-16**, one bit. :mod:`tests.test_adapter`
   * - Fits a drag polar from three solves
     - ``AVLSolver``, trimmed to a lift coefficient -- a different code path through the same physics
     - The fitted polar evaluated at the lift the solver achieved matches the drag it reports to
       **1e-6**, at three lift coefficients the fit never sampled. :mod:`tests.test_adapter`
   * - Claims the polar is exact, not fitted
     - The lattice's own linearity
     - Residual measured at three unsampled angles of attack: **1e-12** or better. If this were a
       surrogate it would have an error budget to publish instead. :mod:`tests.test_adapter`
   * - Reports a span efficiency
     - openavl's ``SPANEF``, read rather than re-derived
     - The identity :math:`k = 1/(\\pi e A\\!R)` holds to **1.6e-15**, which is what says the polar
       is paired with the right two coefficients. :mod:`tests.test_adapter`
   * - Differentiates the lattice w.r.t. the wing
     - ``jax.jacrev`` over ``update_geometry``, the same mechanism ``OpenAVLComp`` uses for its own
       partials, and the subject of openavl's ``reference``-marked ``test_derivatives_three_way``
     - All four wing numbers differenced through the model as installed: **3e-7** (aspect ratio,
       sweep), **1e-5** (taper), and zero asserted for area. :mod:`tests.test_adapter`
   * - Models the 737-800 wing as two untwisted sections
     - ``tests/data/avl/geometries/b737.avl``, openavl's **own reference model of the same
       aeroplane** -- twist, dihedral, a kink, a real airfoil, control surfaces
     - Span efficiency compared and the difference **reported**: cdadt's simplification is
       optimistic. See below. :mod:`tests.test_adapter`

Two vortex lattices on the same wing
------------------------------------

The strongest check on cdadt's aerodynamics is not internal. Each dependency wraps a vortex lattice
-- openavl for cdadt, OpenAeroStruct for OpenConcept -- and the same trapezoidal wing can be put
through both, because ``TrapezoidalPlanformMesh`` is parameterised by the same four numbers
:class:`~cdadt.models.planform.TrapezoidalPlanform` is.

Trimmed to :math:`C_L = 0.5`, with viscous and wave terms switched off on the OpenAeroStruct side so
that both report induced drag alone:

.. list-table::
   :header-rows: 1
   :widths: 55 22 23

   * - Quantity
     - Induced drag
     - Implied e
   * - OpenAeroStruct ``VLM``, near-field
     - 0.008133
     - 1.035
   * - openavl, near-field
     - 0.008079
     - 1.042
   * - openavl, Trefftz far-field (**what cdadt uses**)
     - 0.008533
     - 0.990

**Near-field to near-field the two codes agree to 0.67%**, which is the anchor. Comparing across the
near/far-field boundary instead gives 4.9%, and that is a difference between two *quantities* -- it
is 5.3% within openavl alone.

Since that comparison was worth making once, it is now something a study can make for itself:
:class:`~cdadt.adapter.oas.OpenAeroStructLoads` puts OpenConcept's lattice behind the same slot
:class:`~cdadt.adapter.avl.OpenAVLLoads` sits in, so an aeroplane can be sized on either by changing
one line of a case file. Two independent codes behind one interface is what turns "the lattice says
0.99" from a claim into a measurement. They are **not** interchangeable, and the differences are
declared rather than smoothed over:

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Property
     - ``OpenAVLLoads``
     - ``OpenAeroStructLoads``
   * - Induced drag
     - Trefftz-plane far field
     - near-field panel sum
   * - Span efficiency
     - reported by the code (``SPANEF``)
     - inferred from the fitted curvature
   * - Compressibility
     - Prandtl-Glauert, five Mach samples
     - none; its ``CDi`` is Mach-free
   * - Geometry derivatives
     - ``jax.jacrev``, exact to 3e-7
     - OpenMDAO totals, exact to 3e-7
   * - Polar in :math:`C_L`
     - an identity; residual **1e-12**
     - a fit; residual **2.2e-3**

The last row is the one that would have been easiest to leave out. openavl's quadratic is exact
because the Trefftz drag is a quadratic form in a circulation affine in angle of attack; the
near-field sum carries no such guarantee, and measured at three unsampled angles it drifts 2.2e-3 at
low lift. Both models are evaluated in closed form at every mission node, so that residual is a real
error budget for one of them and round-off for the other.

And note the second column. Both codes' near-field values imply a span efficiency **above 1** for a
planar wing, which is impossible, since elliptical loading is optimal. **OpenConcept's own vortex
lattice has the same limitation, independently.** That is stronger evidence for reading the far field
than cdadt's own reasoning about it was, and it arrived from the other dependency.

What the simplification costs
-----------------------------

This is the one comparison on this page that is not about implementation fidelity, and the only one
cdadt does not pass by being close.

.. list-table::
   :header-rows: 1
   :widths: 54 23 23

   * - Model
     - Aspect ratio
     - e
   * - openavl's own ``b737.avl``
     - 10.13
     - 0.928
   * - cdadt's ``TrapezoidalPlanform``
     - 9.45
     - 0.990

cdadt's wing reports a span efficiency **6.7% above** openavl's own detailed model of the same
aircraft. That is the expected direction -- an untwisted planar trapezoid with no kink is closer to
elliptical loading than a real wing with washout and dihedral -- but the size matters, because the
span efficiency is the entire mechanism by which the lattice changes the aeroplane.

So it was measured rather than left as a caveat. Flying the parabolic polar at each value, which
isolates the span efficiency from everything else, the aeroplane sized at 0.928 instead of 0.990
carries **2.3% more fuel**, weighs 0.6% more, and needs 1.5% more runway. Carried through to the
headline comparison in :doc:`aerodynamics`, the **−8.0% fuel saving becomes about −5.7%** if the
wing is as detailed as openavl's own reference 737 rather than as clean as cdadt's trapezoid.

Read as a bound: **cdadt's lattice result is an optimistic end of a range, not a prediction.** The
test that measures it asserts only that the direction has not inverted and that the gap has not
grown past 15%, because the number is the finding and a tight tolerance would only be a record of
today's geometry.

What the 6.7% is *not* is a defect in how cdadt uses openavl. That was the open question, and the
comparison above settles it: on the identical simplified trapezoid, cdadt's lattice agrees with
OpenConcept's own to 0.67%. So the whole 6.7% is attributable to geometry cdadt does not carry --
twist, dihedral, the kink, a real airfoil -- and not to the lattice being driven wrongly. Carrying
that geometry is a capability cdadt does not have and would need a case file to state; it is scoped
in :doc:`developing` rather than hidden here.

What has no truth reference at all
----------------------------------

Stated here rather than left to be inferred from the absence of a row above.

**The certification layer.** 14 CFR Part 25 is the truth reference for every constraint cdadt
applies, and it is a regulation, not a code. No OpenConcept example computes a certification margin,
so there is nothing to compare against; what can be checked is that each constraint bounds the
quantity it names, in the units it names, and :doc:`certification` lists the requirements cdadt
cannot evaluate at all. Those gaps are in :doc:`validation`.

**Nothing about wave drag any more, and that row was wrong.** It read: *"neither dependency models
it, so there is no truth reference because there is no model."* The first half is false. openavl
indeed cannot, and OpenConcept's jet-transport parasite buildup carries no Mach term -- but
OpenConcept **does** ship a wave drag model, ``WaveDragFromSections``, verified against
OpenAeroStruct by its own tests. It sits under ``aerodynamics/openaerostruct/``, which is why looking
only at the buildup the B738 example uses missed it. cdadt now installs it; see the table above.

The lesson is the one this page exists for: "the dependency has no model for this" is a claim about
the dependency, and it needs checking in the dependency rather than inferring from the one example
that does not use it.

**The optimization result.** No dependency publishes an optimum for this aircraft. What is checked
is that the driver converged, that no constraint is violated, that the objective improved and that
the aeroplane still flies the mission -- properties of a solution, not agreement with a reference.

**Everything cdadt owns that is not physics.** The case-file schema, the run directory, the
artifacts, the discipline ownership map, the command line. These have no dependency analogue, which
is exactly why they are covered by unit and contract tests instead.
