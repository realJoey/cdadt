Optimization
============

:class:`~cdadt.optimization.Optimizer` owns the coupling between what may change, what must
hold, and what is being minimized. All three come out of the case file; none of them is written
in Python.

Order of operations
-------------------

1. **Check everything, before anything expensive.** A cheap probe of the black box is built -- a
   fraction of a second -- and every freed variable is confirmed to be an independent variable
   the box accepts, and the objective and every constraint to be a quantity it publishes. A
   misspelled name is a message naming it with suggestions, not an OpenMDAO error thrown out of
   ``setup`` after a minute of building.
2. **Declare, then build.** Design variables, constraints and the objective are registered on
   the box's own group *before* ``setup``, because OpenMDAO requires it. Registering a design
   variable changes no OpenConcept behaviour; it states which of the box's existing independent
   variables a driver may move.
3. **Converge a baseline.** The continuation ladder is walked once. The driver's first function
   evaluation must begin from a converged aircraft. This is
   :meth:`~cdadt.optimization.Optimizer.prepare`, and it is separable from the run because the
   declared, converged, undriven problem is exactly where total derivatives are checked.
4. **Drive.** Every later design starts from its predecessor's converged state.
5. **Evaluate the basis at the optimum** and report both the design change and the traceability
   matrix.

Freeing a design variable
-------------------------

There is no separate list of design variables. Every ``ac|`` variable is already declared in the
case file's ``design_variables`` block with its value and its source; one becomes free for the
driver by gaining an ``optimize:`` entry *there*:

.. code-block:: yaml

   design_variables:
     ac|geom|wing|S_ref:
       value: 124.6
       units: m**2
       source: b737.org.uk technical specifications
       optimize: {lower: 90.0, upper: 180.0}

     ac|geom|wing|AR:
       value: 9.45
       source: b737.org.uk technical specifications
       optimize: {lower: 7.0, upper: 13.0}

The shipped study frees five that way: wing area, aspect ratio, quarter-chord sweep, taper, and
the engine rating. Deleting the three ``optimize:`` lines of a variable holds it fixed again, and
nothing else in the file changes -- the value, the units and the provenance stay where they were,
so a sizing case and the optimization of the same aeroplane never disagree about a number.

Only independent variables of the box are eligible. Anything the box computes -- operating empty
weight, the tail areas, the maximum lift coefficients, maximum takeoff weight -- is a result of
the design, not a choice within it, and cdadt rejects it by name before the run.

The bounds are an engineering statement. They should be the range over which the empirical
weight and drag correlations inside the box are defensible for the class of aircraft being
designed, not the range over which the code happens to run. An optimum sitting on a bound is
usually telling you the bound is doing the modelling.

Scaling decides the result
--------------------------

Two scalings matter, and neither is decoration.

**Design variables** are scaled by ``ref``, which defaults to the larger of the two bounds. A
wing area of 124 m² and an aspect ratio of 9.45 differ by an order of magnitude; without scaling
the optimizer's trust region is meaningful for one of them and not the other.

**The objective** is scaled by ``ref`` as well, and for SLSQP this is the difference between a
result and a no-op: SciPy's SLSQP takes its finite-difference step and its convergence test on
the *scaled* objective, so a fuel mass of order 1e4 kg is inside tolerance before the first
iteration. ``ref: 2.0e4`` fixes that. IPOPT scales the objective from its own gradient and does
not need it.

**Constraints** are scaled by the magnitude of their own bound, automatically. Without it a climb
gradient in hundredths of a radian is numerically invisible next to a field length in thousands
of feet.

Choosing a driver
-----------------

The code's default is SLSQP, because SciPy is always installed. The shipped case uses IPOPT, and
the reason is specific rather than a preference:

The black box is a Newton-solved implicit system. An optimizer exploring a wide design space
will eventually propose a design that cannot be converged from where the solver currently is,
and the solve raises. OpenMDAO's **pyOptSparse driver catches that**, reports the point as
failed and lets the optimizer shorten its step. OpenMDAO's **SciPy driver re-raises**, so a
single unconvergeable trial design ends the whole run.

For a narrow study -- one or two design variables over a small interval -- SLSQP is fine. For
anything that ranges widely, IPOPT is the right tool, and the failure mode with SLSQP is not
subtle: the run stops with a Newton convergence error rather than producing a wrong answer.

.. code-block:: yaml

   driver:
     name: IPOPT
     maxiter: 40
     tol: 1.0e-6
     derivative_mode: fwd

cdadt supplies sensible per-optimizer defaults for IPOPT (limited-memory Hessian
approximation, adaptive barrier strategy, gradient-based NLP scaling) and SNOPT, and
``driver.options`` overrides any of them.

Derivative mode
---------------

``fwd``, for this class of problem. Totals are taken across the whole Newton-converged coupled
system, and a sizing optimization has a handful of design variables against many responses --
several of which are vectors with one entry per node. Forward mode costs one linear solve per
design variable; reverse mode costs one per response.

Solver settings during an optimization
--------------------------------------

The shipped optimization case raises the Newton iteration limit from 20 to 50. An optimizer
visits designs a human would not, and the coupled solve needs more room at the awkward ones.
``err_on_non_converge`` stays ``true``: with IPOPT, a raised error is information the optimizer
uses. Turning it off would let a half-converged design be scored as if it were real.

What is reported
----------------

Three sections:

**Design variables** -- baseline value, optimum, and the bounds, so an optimum on a bound is
visible rather than inferred.

**Results** -- every scalar response of every discipline, baseline against optimum, with the
percentage change. Not just the objective: an optimization that cut fuel by moving weight
somewhere unacceptable should be visible in the same table.

**The certification basis** -- the full traceability matrix at the optimum. See
:doc:`certification`.

An optimization is reported as having succeeded only if the driver converged **and** every
constraint is satisfied. Both halves matter: a driver that stops on its iteration limit inside
an infeasible region has not solved the problem, and a report that calls that an optimum is
wrong. ``cdadt optimize`` exits non-zero in that case, so a shell script cannot archive a
failure as a success.

The shipped study
-----------------

Minimizing fuel with reserves over the wing planform and the engine rating, subject to 14 CFR
25.113, 25.121(b)(1)(i) and the engine deck's throttle band in climb and cruise. IPOPT, 38
objective evaluations and 23 sensitivity evaluations, about two minutes at 11 nodes per phase:

=============================  ===========  ===========  ==========
Quantity                       Baseline     Optimum      Change
=============================  ===========  ===========  ==========
Fuel with reserves (kg)        18,596.8     16,399.6     **-11.8%**
Maximum takeoff weight (kg)    78,345.0     72,324.2     -7.7%
Engine rating (lbf)            27,000       21,357.8     -20.9%
Balanced field length (ft)     5,247.7      6,228.7      +18.7%
=============================  ===========  ===========  ==========

Four of four constraints met, one active: the **climb throttle band**. Neither certification
constraint binds, so the design that comes out is not, in this study, a certification-limited
design. It is limited by the engine deck running out of throttle in the climb.

.. note::

   **These numbers moved, and the reason is worth keeping.** An earlier version of this study
   allowed a wing area down to 90 m2, which at the weights an optimizer reaches puts the wing
   loading at 759 kg/m2 -- outside the 500-750 band :doc:`validation` publishes as the range the
   box's empirical weight correlations were fitted over, and a region where its Newton solver
   simply fails to converge. It reported -14.4% on fuel. That was a better answer obtained by
   walking somewhere the model is not valid, and the bounds now keep the driver inside it.

.. important::

   **Three of the five design variables end on a bound.** Aspect ratio sits on its upper bound of
   13, and quarter-chord sweep and taper on their lower bounds of 15° and 0.12. That is the
   bounds doing the modelling, and it has to be read as such: the study says the box's drag and
   weight correlations would keep paying for more span, less sweep and more taper as far as it is
   willing to extrapolate them. Widening the intervals would move the answer. Whether they
   *should* be widened is an engineering judgement about where those correlations remain
   defensible for a transport-category aeroplane, not a numerical one.

Reproduce it with:

.. code-block:: bash

   cdadt optimize cases/b738_optimization.yaml

Read :doc:`validation` before quoting these numbers. In particular the engine-out gradient is
evaluated clean rather than in the takeoff configuration §25.121(b) specifies, and the engine is
a scaled deck rather than a redesigned engine -- a 21% reduction in rating is a long way down the
surrogate.

The same study on a vortex lattice
-----------------------------------

Three optimizations ship, and they differ in one thing: where the induced drag comes from. Same
aeroplane, same mission, same reserves, same five design variables, same bounds, same certification
basis, same driver.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * -
     - reference
     - openavl
     - OpenAeroStruct
   * - Fuel with reserves (kg)
     - 16,399.6
     - 15,314.9
     - 15,066.7
   * - against its own baseline
     - −11.8%
     - −12.0%
     - −12.3%
   * - Maximum takeoff weight (kg)
     - 72,324.2
     - 71,642.9
     - 71,265.6
   * - Engine rating (lbf)
     - 21,357.8
     - 19,985.5
     - 19,677.1
   * - **Quarter-chord sweep (deg)**
     - **15.0 — lower bound**
     - **31.41**
     - **31.56**
   * - Taper ratio
     - 0.1345
     - 0.1598
     - 0.2358
   * - Wing span (m)
     - not published
     - 35.50
     - 35.50
   * - Constraints
     - 4/4 met, 1 active
     - 5/5 met, 1 active
     - 5/5 met, 1 active

**Sweep is the row that matters, and it is not the lattice's doing.** The reference drives
quarter-chord sweep to its lower bound because in that model sweep can only cost -- it adds
structural weight and buys nothing, since OpenConcept's B738 group has no transonic drag rise
anywhere and cannot be given any. Turn drag rise on and sweep goes the other way, to about 31
degrees, stopping short of its 32-degree bound rather than pinning to it: an interior optimum, which
is what a correctly posed trade looks like.

Attributing that to the vortex lattice would be wrong, and it was a mistake made once here. Flying
the *parabolic polar* with wave drag on puts sweep at 31.1 degrees too. Sweep is driven by the
drag-rise model, which depends on sweep whatever computes the induced drag. What the lattices
contribute is the fuel, not the sweep.

**The two independent codes land within 0.15 degrees of each other** on sweep and within 1.6% on
fuel, which is the strongest cross-check in the repository: two vortex lattices, wrapped by
different projects, given the same wing.

They disagree most on **taper** -- 0.16 against 0.24 -- and that is the honest residual. Taper is
where a near-field induced drag and a Trefftz-plane one differ most, and :doc:`truth` gives the
measured size of that difference rather than averaging it away.

What is *not* shown here: an aeroplane whose sweep is credible in absolute terms. cdadt's wing is
two untwisted sections, which is about 6.7% optimistic on span efficiency against openavl's own
detailed 737 model, and the Korn drag rise is a correlation rather than a transonic solve. The
result is a coherent trade between models that are each documented, not a prediction.

Reproduce them with:

.. code-block:: bash

   cdadt optimize cases/b738_avl_optimization.yaml    # needs the [avl] extra
   cdadt optimize cases/b738_oas_optimization.yaml    # needs the [transonic] extra

When bounds are not preferences
--------------------------------

One of these studies failed before it worked, and the reason is worth keeping.

An earlier version allowed wing area down to 90 m². At the weights an optimizer reaches, that is a
wing loading of 759 kg/m² -- outside the 500-750 band :doc:`validation` publishes as the range the
box's empirical weight correlations were fitted over. IPOPT walked there, the Newton solver failed
to converge in 50 iterations, and the run exited on *"invalid number in NLP function or derivative
detected"*.

It read like an aerodynamics bug and was not: at that design the box fails identically with the
parabolic polar, and the lattice's values and gradients are finite at every bound. The design space
was simply larger than the model it describes.

The repair was not a looser tolerance. It was bounds that keep the driver inside the envelope, and
a **span constraint** in place of one that had been standing in for it. OpenConcept's own optimizing
example writes ``add_design_var("ac|geom|wing|AR", lower=5.0, upper=10.4)  # limit to fit in group
III gate`` -- an aspect-ratio bound expressing a span limit, which works only because that example
holds the area fixed. A sizing study frees the area, so the limit has to be said as what it is:

.. code-block:: yaml

   - name: wing_span
     upper: 36.0
     units: m
     source: FAA Airplane Design Group III gate limit, 118 ft wingspan

The published optimum moved as a result, from −14.4% to −11.8% on fuel. The old figure was better
because it was obtained somewhere the model is not valid, which is the least useful kind of better.
