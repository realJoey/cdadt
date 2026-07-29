Supplying your own aerodynamics
===============================

cdadt drives OpenConcept as a black box, and for everything except the aerodynamics it still
does. The drag can be **cdadt's own** -- computed from a vortex lattice built on the wing the case
file describes -- while the trajectory, the balanced field, the reserves, the engine deck and the
weight closure remain OpenConcept's, used as published.

Switching between them is one line of a case file:

.. code-block:: yaml

   black_box:
     model: cdadt.adapter.analysis:SizingMissionAnalysis
     options:
       aerodynamic_loads: cdadt.adapter.avl:OpenAVLLoads    # or cdadt.models.polar:PolarLoads
     num_nodes: 21

``cases/b738_cdadt_aero.yaml`` is that case, and it is the same aeroplane and the same mission as
``cases/b738.yaml``.

.. contents::
   :local:
   :depth: 1

Why cdadt needs its own analysis group
--------------------------------------

OpenConcept was built for this. Every mission profile declares the aircraft model as an option:

.. code-block:: python

   # openconcept/mission/profiles.py
   self.options.declare("aircraft_model", default=None, desc="OpenConcept-compliant airplane model")

and ``openconcept/examples/B738_VLM_drag.py`` uses exactly that hook to swap in a vortex-lattice
drag model. The only obstacle is that ``B738_sizing.py`` writes
``aircraft_model=B738AircraftModel`` at the line where it builds the mission, so the choice cannot
be reached from outside that group.

So cdadt owns an aircraft model and the analysis group that installs it.
:class:`~cdadt.adapter.analysis.SizingMissionAnalysis` composes the same geometry, maximum-lift,
empty-weight and mission pieces OpenConcept's own example composes -- unmodified -- and differs in
two deliberate ways: the aeroplane comes from the case file rather than a data dictionary inside
the library, and the aerodynamics is a parameter.

**Nothing in either dependency is modified, copied, subclassed or patched.** Five contract tests
hold that line, and :mod:`cdadt.adapter` is the one package permitted to import a dependency at
all -- which is what the project's brief asks for. A sixth test holds the other half: the physics
in :mod:`cdadt.models` imports neither, so a model stays writable and testable without them.

The layers
----------

.. code-block:: text

   cdadt.models              physics cdadt owns. OpenMDAO and numpy only.
     AerodynamicLoads          the abstraction -- produces six coefficients
     PolarLoads                a parabolic polar
     TrapezoidalPlanform       the case file's four numbers -> lattice sections
        |
   cdadt.adapter             the only package importing a dependency
     AerodynamicLoadsComp      evaluates a model over a phase, publishes `drag`
     OpenAVLLoads              an AerodynamicLoads backed by openavl
     CdadtAircraftModel        cdadt's drag + OpenConcept's engine and weights
     SizingMissionAnalysis     installs it into FullMissionWithReserve
        |
   OpenConcept               the mission, unchanged

A model is handed a :class:`~cdadt.models.loads.FlightCondition` and a
:class:`~cdadt.models.loads.Planform` and returns
:class:`~cdadt.models.coefficients.AeroCoefficients`. It never sees OpenMDAO or OpenConcept. The
interface carries all six components -- lift, drag, sideforce and three moments -- even though the
mission consumes only the drag, because that is what a lattice returns for free and what trim,
static margin and handling qualities are written in.

The shape follows falco's :class:`falco.core.loads.loads.Loads`, which declares one abstract
method turning a flight state into forces and moments.

Writing a model
---------------

.. code-block:: python

   from cdadt.models import AeroCoefficients, AerodynamicLoads

   class MyLoads(AerodynamicLoads):
       """Whatever aerodynamics you want."""

       model_name = "my_model"          # abstract: no model can inherit its own name

       @classmethod
       def build(cls, *, planform, span_efficiency, zero_lift_drag):
           """Say which of the box's values this model consumes."""
           return cls(...)

       def coefficients(self, condition, planform):
           """Return six coefficients at every point of ``condition``."""
           return AeroCoefficients(CL=condition.CL, CD=...)

       def drag_gradients(self, condition, planform):
           """Analytic derivatives. An optimizer steps on these."""
           return {"CL": ..., "CD0": ..., "e": ..., "AR": ...}

Then name it in a case file. Nothing else changes -- not the disciplines, not the optimizer, not
the command line.

Two things are not optional. **Derivatives must be analytic**: these components are real physics
now, an optimizer differentiates them, and :doc:`verification` publishes a derivative study.
**Construction must be cheap**: :meth:`~cdadt.models.loads.AerodynamicLoads.build` is called on
every evaluation, because the span efficiency and the zero-lift drag are variables of the black
box that move under a driver. Anything expensive belongs behind a cache keyed on what it depends
on, as the lattice is.

The vortex lattice
------------------

:class:`~cdadt.adapter.avl.OpenAVLLoads` builds the case file's wing as an AVL lattice through
openavl and takes its drag from it.

**It is not solved at every node.** A sizing mission is about 170 analysis points inside a Newton
solve inside an optimizer; openavl's OpenMDAO component evaluates one flight point per instance.
OpenConcept hit the same wall and answered it with a trained surrogate -- its ``VLMDragPolar``
exists because *"a surrogate model to decrease the computational cost"* was necessary.

cdadt takes a cheaper and exact route, because a vortex lattice is **linear**. For a fixed
geometry the induced drag is quadratic in lift, so three solves determine the whole polar:

.. math::

   C_D = C_{D_\\mathrm{min}} + k\\,(C_L - C_{L_\\mathrm{minD}})^2

and every node is then evaluated in closed form. The fit is re-derived whenever the planform
moves, so it stays exact under an optimizer rather than drifting like a surrogate. Checked against
direct lattice solves at lift coefficients the fit never sampled, it agrees to about 0.1% across
the mission range.

Two results from the shipped 737-800 wing, and one caution
-----------------------------------------------------------

The lattice reports a span efficiency of **0.990** where ``cases/b738.yaml`` assumes
``ac|aero|polar|e = 0.82``. Flying the mission on it produces a visibly different aeroplane:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - Parabolic polar
     - Vortex lattice
     - Change
   * - Maximum takeoff weight (kg)
     - 78,345.0
     - 76,540.5
     - −2.3%
   * - Operating empty weight (kg)
     - 41,748.2
     - 41,363.0
     - −0.9%
   * - Fuel with reserves (kg)
     - 18,596.8
     - 17,177.6
     - −7.6%
   * - Balanced field length (ft)
     - 5,247.7
     - 4,952.4
     - −5.6%

The chain is coherent: less induced drag, so less fuel, so a lighter aeroplane, so a shorter
field. **It is not a validated result.** It says what this lattice predicts for this planform, and
the lattice sees a wing alone -- no fuselage, no tails, no nacelles, and no compressibility beyond
a Prandtl-Glauert correction. The parasite drag is still OpenConcept's component buildup. Read
:doc:`validation` before quoting any of it.

.. warning::

   **Read the far-field drag, not the near-field.** openavl reports both, and this model got it
   wrong first: near-field ``CD`` underestimates induced drag on a swept wing badly enough to
   give a span efficiency of 1.06 for the shipped planform -- impossible, since a planar wing
   cannot beat elliptical loading. ``CDFF``, the Trefftz-plane value, gives 0.99, and openavl's
   own ``SPANEF`` agrees.

   The setup was confirmed correct by the case theory pins down: a rectangular wing of the same
   aspect ratio comes out at **0.965**, a little under 1 exactly as it must.

What the geometry derivative does and does not include
-------------------------------------------------------

The lift and zero-lift-drag derivatives are exact. The aspect-ratio derivative holds the
lattice's span efficiency fixed, giving
:math:`\\partial C_D/\\partial A\\!R = -k(C_L-C_{L_0})^2/A\\!R`. The exact derivative carries a
second term in :math:`\\partial e/\\partial A\\!R`, which for this wing is
:math:`-1.85\\times10^{-3}` -- **1.8%** of the term that is kept, so this captures 98% of the
sensitivity. ``test_the_neglected_span_efficiency_gradient_stays_small`` measures that rather than
assuming it stays small.

Making it exact means chaining openavl's ``jacrev`` over its ``GeometryDesignParams`` with the
analytic section derivatives of :class:`~cdadt.models.planform.TrapezoidalPlanform`, and letting
the reference area move with the wing, which ``snapshot_refs`` currently holds fixed. That is
outstanding work, not a line of it.

Installing openavl
------------------

openavl is an **optional extra**. Everything else in cdadt works without it, and the tests that
need it skip rather than fail.

.. code-block:: bash

   pip install -e ".[avl]"

One caution, because it will otherwise break a working environment: openavl declares
``numpy>=2.4``, and cdadt is pinned to ``numpy<2`` for the reason :doc:`install` gives -- with
MKL-backed BLAS, NumPy aborts the interpreter inside OpenConcept's import. Installing openavl's
dependencies unpinned pulls numpy 2.4 over the pinned 1.26. Its declared floor turns out not to be
load-bearing, so install it without its dependencies and take a JAX that keeps the pin:

.. code-block:: bash

   pip install "jax<0.5" "jaxlib<0.5"
   pip install -e /path/to/openavl --no-deps
