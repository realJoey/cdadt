Supplying your own aerodynamics
===============================

cdadt drives OpenConcept as a black box, and for everything except the aerodynamics it still
does. The drag can be **cdadt's own** -- computed from a vortex lattice built on the wing the case
file describes -- while the trajectory, the balanced field, the reserves, the engine deck and the
weight closure remain OpenConcept's, used as published.

**Three models ship**, and they are chosen by one line: a parabolic polar, an openavl vortex
lattice, and OpenConcept's own OpenAeroStruct lattice. The last two solve the same wing with
independent codes, which is what makes cdadt's aerodynamics a checkable claim rather than one
solver's word for it -- see :doc:`truth` for how far they agree and exactly where they do not.

Switching between them is one line of a case file:

.. code-block:: yaml

   black_box:
     model: cdadt.adapter.analysis:SizingMissionAnalysis
     options:
       aerodynamic_loads: cdadt.adapter.avl:OpenAVLLoads
       #                  cdadt.adapter.oas:OpenAeroStructLoads
       #                  cdadt.models.polar:PolarLoads
     num_nodes: 21

.. contents::
   :local:
   :depth: 1

The shipped cases, and which question each answers
---------------------------------------------------

Every one flies the same aeroplane, the same 2800 nmi mission and the same reserves. They differ in
where the drag comes from, and in one case in whether the aeroplane is flown physically completely.

.. list-table::
   :header-rows: 1
   :widths: 30 22 12 36

   * - Case
     - Aerodynamics
     - Wave drag
     - What it is for
   * - ``b738.yaml``
     - OpenConcept's own
     - none possible
     - The reference. Drives ``B738SizingMissionAnalysis`` itself
   * - ``b738_avl.yaml``
     - openavl lattice
     - on
     - Induced drag from the wing's shape, far-field
   * - ``b738_oas.yaml``
     - OpenAeroStruct lattice
     - on
     - The same, through OpenConcept's own lattice, near-field

Each has an ``_optimization`` twin -- ``b738_optimization.yaml``,
``b738_avl_optimization.yaml`` and ``b738_oas_optimization.yaml``. Three sets of two: the aircraft
configuration driven into the black box, and the same configuration with each of the two vortex
lattices supplying the aerodynamic loads.

The parity anchor is not among them, and deliberately so. :mod:`tests.test_adapter` builds it at
run time from ``b738.yaml`` -- cdadt's analysis group, the parabolic polar, wave drag off -- and
asserts it reproduces the reference to 4e-13. A verification case belongs where it is checked on
every run, not in a directory where it depends on somebody remembering to run it. The twins differ from their sizing cases in three things and no others:
eleven nodes rather than twenty-one, ``optimize:`` entries on five design variables, and the
``driver``, ``constraints`` and ``objective`` blocks at the end.

One caution about comparing against the reference, because it is the difference between a number
that means something and one that does not. The reference has no transonic drag rise anywhere and
cannot be given any, so a lattice measured against it nets an induced-drag saving against a
drag-rise penalty and reports the two as one figure.

Separating them takes one line: fly ``cdadt.models.polar:PolarLoads`` with ``wave_drag: true``, which
shares the compressibility and differs from a lattice case only in where the induced drag comes from.
Measured that way, the drag rise costs **+1.8%** of fuel and the lattice is worth **-8.1%**, against
the -6.4% the two together show. No such case is shipped -- it answers a question about the models
rather than about an aeroplane -- but it is worth running before quoting either number alone.

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
     LatticeSolver             a vortex-lattice code, reduced to `fit() -> LatticePolar`
       DifferentiableLattice     openavl
       OpenAeroStructLattice     OpenConcept's OpenAeroStruct lattice
     LatticeLibrary            solved polars; one per study, injected, never global
     OpenAVLLoads              an AerodynamicLoads backed by openavl
     OpenAeroStructLoads       an AerodynamicLoads backed by OpenAeroStruct
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

Why lift goes *in* and only drag comes *out*
---------------------------------------------

The single most surprising thing about this interface is that the aerodynamics is handed the lift
coefficient rather than producing it. It is worth understanding, because it explains why five of the
six coefficients a lattice computes have nowhere to go.

**OpenConcept's mission is a point-mass trajectory.** Its state is position, speed and weight. It
has no attitude, and ``alpha`` appears **zero times** in ``openconcept/mission/phases.py`` and
``openconcept/mission/profiles.py`` -- that is a count, not an impression. With no angle of attack
there is no variable an aerodynamic model could solve *for*, so lift cannot be an aerodynamic result.

Instead it is a **kinematic requirement**. ``SteadyFlightCL`` states vertical equilibrium and solves
it for the lift coefficient:

.. math::

   C_L = \\frac{\\cos\\gamma \\; g \\, W}{q \\, S_\\mathrm{ref}}

Every quantity on the right is something the trajectory already knows. The aircraft is holding a
flight path at a weight and a speed, so the lift it is producing is not in question -- it is
determined. What *is* in question is what that lift costs, and that is the one thing the
aerodynamics is asked.

.. code-block:: text

   the mission knows                      cdadt's model answers
   ------------------                     ---------------------
   weight W  ---------.
   dynamic pressure q -+--> CL = cosY g W  --> [ AerodynamicLoads ] --> CD --> drag = CD q S
   wing area S -------'         q S              polar / lattice                    |
   flight path angle Y                                                              v
        ^                                                                      thrust = drag
        |                                                                      fuel flow
        '--------------------- weight falls as fuel burns <----------------------- '

The loop closes through weight, not through lift: more drag means more fuel, which means a lighter
aircraft later and a heavier one at takeoff, which changes the lift required. That is the coupling
the Newton solver resolves.

**Two phases do not use equilibrium at all**, and they are worth knowing about because they are
places the box asserts a lift coefficient outright:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Phase
     - :math:`C_L` comes from
     - What that means
   * - climb, cruise, descent, reserve, loiter
     - ``SteadyFlightCL``
     - Vertical equilibrium, as above
   * - rotate
     - ``CL_rotate_mult`` × ``ac|aero|CLmax_TO``
     - A prescribed fraction of maximum lift, not a solve
   * - ground roll (``v0v1``, ``v1v0``, ``v1vr``)
     - a constant, ``0.1``
     - *"set CL = 0.1 for the ground roll per Raymer's book"* -- hardcoded inside the box and not
       reachable from a case file

So the aerodynamic model is consulted in all of them, but in the last two it is answering a question
about a lift coefficient nobody solved for.

What would have to change for lift to be an input
--------------------------------------------------

Feeding lift *into* the box is not a matter of connecting one more variable. It would
**over-determine** the system: the mission already states :math:`C_L` from equilibrium, a lattice
would state :math:`C_L(\\alpha)` from the flow, and with no free variable between them the two
statements simply contradict each other unless they happen to agree.

The free variable that is missing is **angle of attack**. Introduce it as an unknown, add the
residual

.. math::

   C_{L_\\mathrm{lattice}}(\\alpha) - C_{L_\\mathrm{required}} = 0

and lift becomes a genuine coupling rather than an assumption -- the solver picks the attitude that
makes the wing produce the lift the trajectory needs. **That is what a trim solve is**, and it is the
point at which the other five coefficients start to matter: with an :math:`\\alpha`, a centre of
gravity and a tail, :math:`C_m` becomes a second residual and the aircraft is trimmed rather than
merely balanced vertically.

None of that exists in the box today, and cdadt does not pretend otherwise. Both lattices are
*parameterised* by :math:`\\alpha` internally -- they are solved at three angles -- but the fit
inverts that relationship, producing :math:`C_D(C_L)` so the model can answer the only question the
mission asks without ever needing an attitude. The inversion is the reason cdadt's aerodynamics
drops into a point-mass mission at all.

What each model produces, and what is consumed
-----------------------------------------------

.. code-block:: text

                        produced by     produced by     consumed by
                        PolarLoads      the lattices    the mission
   ---------------------------------------------------------------
   CL   lift               echoed          echoed          input, not output
   CD   drag                 yes             yes           YES -> drag force
   CY   sideforce             no             yes*           no consumer
   Cl   roll                  no             yes*           no consumer
   Cm   pitch                 no             yes*           no consumer
   Cn   yaw                   no             yes*           no consumer

   * openavl computes all six; cdadt currently reads the forces it fits a polar from.
     OpenAeroStruct's VLM publishes CL, CDi, CDv, CDw and CM.

:class:`~cdadt.models.coefficients.AeroCoefficients` carries all six deliberately, even though the
mission consumes one. The interface is shaped for what a lattice *is* rather than for what today's
black box happens to want, so a trim residual, a static-margin constraint or a handling-qualities
study is a new consumer rather than a new interface. Reporting a fabricated zero moment would be a
different claim from reporting that a model does not produce one, which is what
:attr:`~cdadt.models.coefficients.AeroCoefficients.lateral_directional` exists to say.

Writing a model
---------------

.. code-block:: python

   from cdadt.models import AeroCoefficients, AerodynamicLoads

   class MyLoads(AerodynamicLoads):
       """Whatever aerodynamics you want."""

       model_name = "my_model"          # abstract: no model can inherit its own name

       @classmethod
       def build(cls, *, planform, span_efficiency, zero_lift_drag, workspace=None):
           """Say which of the box's values this model consumes."""
           return cls(...)

       def coefficients(self, condition, planform):
           """Return six coefficients at every point of ``condition``."""
           return AeroCoefficients(CL=condition.CL, CD=...)

       def drag_gradients(self, condition, planform):
           """Analytic derivatives, one per GRADIENT_NAMES. An optimizer steps on these."""
           return {"CL": ..., "CD0": ..., "e": ..., "area": ..., "AR": ..., "sweep": ..., "taper": ...}

       DRAG_DEPENDS_ON = ("CL", "CD0", "e", "AR")   # optional: which of them can be non-zero

Then name it in a case file. Nothing else changes -- not the disciplines, not the optimizer, not
the command line.

Every name in :attr:`~cdadt.models.loads.AerodynamicLoads.GRADIENT_NAMES` must be answered, zeros
included. A variable a model does not read is a statement the model makes -- a parabolic polar
genuinely has no taper derivative -- and saying so explicitly is what lets one wrapper install any
model. ``DRAG_DEPENDS_ON`` narrows which of them the wrapper declares partials for, so the sparsity
OpenMDAO is given is the truth rather than a superset.

Two things are not optional. **Derivatives must be analytic**: these components are real physics
now, an optimizer differentiates them, and :doc:`verification` publishes a derivative study.
**Construction must be cheap**: :meth:`~cdadt.models.loads.AerodynamicLoads.build` is called on
every evaluation, because the span efficiency and the zero-lift drag are variables of the black box
that move under a driver.

A model that cannot be cheap keeps what it needs in the ``workspace`` its caller injects. That is
how both lattices afford to be exact: a solve costs tens of seconds, the results live in a
:class:`~cdadt.adapter.lattice.LatticeLibrary` the analysis group owns for the length of a study,
and constructing the model costs nothing. A module-level cache would do the same job and is what
this used to be -- but it is a global by another name, it outlives every caller, and the project's
brief rules it out. A contract test now fails on ``lru_cache`` anywhere in cdadt.

The vortex lattice
------------------

:class:`~cdadt.adapter.avl.OpenAVLLoads` builds the case file's wing as an AVL lattice through
openavl and takes its drag from it.

**It is not solved at every node.** A sizing mission is about 170 analysis points inside a Newton
solve inside an optimizer; openavl's OpenMDAO component evaluates one flight point per instance.
OpenConcept hit the same wall and answered it with a trained surrogate -- its ``VLMDragPolar``
exists because *"a surrogate model to decrease the computational cost"* was necessary.

cdadt takes a cheaper and exact route, because a vortex lattice is **linear**. The circulation is
affine in angle of attack, so the far-field lift is linear in it and the far-field drag is a
quadratic form in the circulation. For a fixed geometry, therefore,

.. math::

   C_D = C_{D_\\mathrm{min}} + k\\,(C_L - C_{L_\\mathrm{minD}})^2

is an **identity rather than a curve fit**, and three solves determine it exactly. Every node is
then evaluated in closed form, and the polar is re-derived whenever the planform moves, so it stays
exact under an optimizer rather than drifting like a surrogate.

The word *exactly* is measured, not asserted. Evaluated at three angles of attack the fit never
sampled, the closed form reproduces the lattice to **1e-12 relative** or better
(``test_the_lattice_polar_is_exactly_quadratic``), and cross-checked against openavl's own
``AVLSolver`` -- a different code path, the one its reference tests validate against the Fortran
binaries -- the two agree to better than **1e-6**
(``test_the_polar_agrees_with_openavls_own_solver``).

An offset is carried rather than writing :math:`C_D \\propto C_L^2` because twist and camber move
the minimum-drag point away from zero lift. For the shipped untwisted, uncambered wing openavl
returns :math:`C_{D_\\mathrm{min}}` and :math:`C_{L_\\mathrm{minD}}` of exactly zero, to 1e-33 --
which is worth knowing, because an earlier version reported 5.6e-5 and 0.0063 there, and those
were the fit absorbing an inconsistency rather than physics.

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
     - 78,345.6
     - 76,827.2
     - −1.9%
   * - Operating empty weight (kg)
     - 41,748.3
     - 41,346.4
     - −0.7%
   * - Fuel with reserves (kg)
     - 18,597.3
     - 17,402.7
     - −6.4%
   * - Balanced field length (ft)
     - 5,247.8
     - 4,986.0
     - −5.0%

The parabolic-polar column is the validated reference to all printed digits, which is the point of
having it: the lattice column is the only thing that changed.

The chain is coherent: less induced drag, so less fuel, so a lighter aeroplane, so a shorter
field. **It is not a validated result.** It says what this lattice predicts for this planform, and
the lattice sees a wing alone -- no fuselage, no tails, no nacelles. The parasite drag is still
OpenConcept's component buildup, and the transonic drag rise is off in this comparison, as it is by
default.

It is also the **optimistic end of a range**, and by a measured amount. openavl ships its own
detailed model of this aircraft, ``b737.avl``, with twist, dihedral, a kink and a real airfoil; it
reports a span efficiency of 0.928 where cdadt's two-section untwisted trapezoid reports 0.990.
Sized on 0.928 instead, the lattice saving shrinks by roughly a quarter. Read :doc:`truth` for
that comparison and :doc:`validation` before quoting any of it.

Across Mach, the lattice's span efficiency rises from 0.990 at M 0 to 0.998 at M 0.85 and the
curvature falls 0.8%, which is the whole of the compressibility effect it can see -- a
Prandtl-Glauert scaling, and no drag rise.

.. warning::

   **Read the far-field pair, and read both halves of it.** openavl reports induced drag twice --
   a near-field pressure sum (``CD``) and a Trefftz-plane far-field value (``CDFF``) -- and lift
   twice with it. This model got the choice wrong twice over, and each mistake looked plausible:

   #. Taking near-field ``CD`` underestimates induced drag on a swept wing badly enough to give a
      span efficiency of **1.06** for the shipped planform -- impossible, since a planar wing
      cannot beat elliptical loading.
   #. Taking far-field ``CDFF`` but pairing it with the *commanded* lift the solver was trimmed to,
      which is a near-field quantity, biased the curvature by **2.2%**. Nothing looked wrong: the
      model reported ``e`` = 0.990 while its own drag implied 0.969, and no test compared the two.

   The pairing that is right is ``CDFF`` with ``CLFF``, and it is right for a checkable reason: it
   satisfies :math:`C_{D_\\mathrm{FF}} = C_{L_\\mathrm{FF}}^2/(\\pi e A\\!R)` with openavl's own
   ``SPANEF`` to **1.6e-15**. ``test_the_fitted_curvature_is_openavls_own_span_efficiency`` is that
   check, and it is the one that makes this class of mistake impossible to reintroduce quietly.

   The setup was confirmed correct by the case theory pins down: a rectangular wing of the same
   aspect ratio comes out just under 1, exactly as it must.

The geometry derivatives are exact
----------------------------------

Every derivative this model reports is analytic and exact, including with respect to the wing:

.. math::

   \\frac{\\partial C_D}{\\partial g} =
       \\frac{\\partial C_{D_\\mathrm{min}}}{\\partial g}
     + \\frac{\\partial k}{\\partial g}\\,(C_L - C_{L_\\mathrm{minD}})^2
     - 2k\\,(C_L - C_{L_\\mathrm{minD}})\\,\\frac{\\partial C_{L_\\mathrm{minD}}}{\\partial g}

for :math:`g` in area, aspect ratio, quarter-chord sweep and taper ratio. The three polar
derivatives come from ``jax.jacrev`` over openavl's own ``update_geometry``, which rebuilds the
panels inside JAX from those four numbers -- so what is differentiated is the lattice, not a formula
about it. Reverse mode rather than forward, because openavl registers its circulation solve as a
``custom_vjp`` and ``jacfwd`` raises on it; a single :func:`jax.vjp` gives the coefficients and
three reverse passes gives their Jacobian, from one set of solves.

Two earlier defects are worth recording, because both were invisible:

- **The aspect-ratio derivative was approximate.** Writing :math:`k = 1/(\\pi e A\\!R)` and holding
  the span efficiency fixed drops a term in :math:`\\partial e/\\partial A\\!R` worth
  :math:`-1.85\\times10^{-3}`, which is **1.8%** of the term kept.
- **Sweep and taper had no derivative at all.** They were inputs with no declared partial, which
  OpenMDAO reads as exactly zero -- so an optimizer was told that changing the taper of the wing
  does not change its drag. The taper term is comparable in magnitude to the aspect-ratio one.

``test_the_geometry_derivatives_are_exact_not_approximate`` differences all four through
:meth:`~cdadt.adapter.avl.OpenAVLLoads.coefficients`, so what is checked is the derivative of the
number the mission consumes, Mach interpolation included. The agreement is 3e-7 for aspect ratio
and sweep and 1e-5 for taper. The area derivative of the drag *coefficient* is zero and asserted to
be zero: a coefficient is normalised by the area it is computed on. The force still depends on area,
and :class:`~cdadt.adapter.loads.AerodynamicLoadsComp` supplies that term.

Transonic drag rise, from OpenConcept
-------------------------------------

A vortex lattice cannot produce shock drag and OpenConcept's jet-transport parasite buildup carries
no Mach term, so for a while cdadt had no transonic drag rise at all and said so. That was the wrong
conclusion from the right observation: OpenConcept ships ``WaveDragFromSections``, a Korn-equation
model *"based on the Korn equation"* using *"the same wave drag approximation as OpenAeroStruct"*,
verified against OpenAeroStruct by OpenConcept's own ``test_wave_drag.py``. It is under
``aerodynamics/openaerostruct/``, which is why reading only the buildup the B738 example uses missed
it.

.. code-block:: yaml

   black_box:
     options:
       wave_drag: true

cdadt supplies the sections from its planform -- :class:`~cdadt.adapter.sections.WingSectionsComp`,
with analytic derivatives -- and adds the result to the parasite drag.

**The shipped mission cruises at M 0.7854.** That is not a number the case file states; it falls out
of the altitude and airspeed it does state, and it is held for the whole cruise phase. At that Mach,
on this wing at the 0.12 thickness the case declares, the Korn model puts wave drag at **2.2% of
total cruise drag**, rising to **11.6%** at the M 0.82 declared as ``Mach_max``:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Mach
     - :math:`C_{D_\mathrm{wave}}`
     - share of total drag
   * - 0.70
     - 0.000000
     - 0.00%
   * - 0.75
     - 0.000022
     - 0.08%
   * - **0.7854 (cruise)**
     - **0.000609**
     - **2.20%**
   * - 0.80
     - 0.001414
     - 4.97%
   * - 0.82 (``Mach_max``)
     - 0.003557
     - 11.62%

So it is **on by default** in every case file that flies a lattice. Neither the lattice nor
OpenConcept's parasite buildup carries a Mach term, so with it off a transonic aeroplane is flown
with no drag rise at all -- and the sweep the optimizer picks is meaningless, as :doc:`optimization`
shows.

The exception is the parity anchor: ``B738AircraftModel`` has no wave drag, so reproducing the
reference example requires it off. That is a verification case, not a design case.

These figures were first published an order of magnitude too small, from a probe that used a
thickness ratio of 0.10 against the case file's 0.12. A drag-rise model is exponential in the wrong
direction to guess at.

What it costs
-------------

About **6 seconds per wing per Mach sample** -- roughly 30 seconds per distinct geometry, cached on
the four wing numbers, so a mission pays it once and an optimizer once per design iteration. That
buys exact derivatives instead of a surrogate's; it is the price of solving the lattice rather than
interpolating a table of it.

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

Installing OpenAeroStruct
-------------------------

Needed for the transonic drag rise above, and for the validation that compares cdadt's lattice
against OpenConcept's own. It is where OpenConcept keeps both components, so cdadt reaches it only
through OpenConcept and never imports it outside a test.

.. code-block:: bash

   pip install -e ".[transonic]"          # or: pip install -e /path/to/OpenAeroStruct

Unlike openavl this one is safe to install with its dependencies. Its requirements are
``openmdao>=3.35,!=3.40``, ``numpy>=1.21``, ``scipy>=1.7`` and ``matplotlib`` -- all floors, no
ceilings -- so the pinned numpy 1.26 already satisfies them and nothing is upgraded. Verify with a
dry run anyway, since that is cheap and the failure it prevents is an interpreter abort:

.. code-block:: bash

   pip install --dry-run -e /path/to/OpenAeroStruct   # expect: "Would install openaerostruct" alone
