Optimization
============

:class:`~cdadt.optimization.Optimizer` owns the coupling between what may change, what must
hold, and what is being minimized. All three come out of the case file; none of them is written
in Python.

Order of operations
-------------------

1. **Check everything, before anything expensive.** A three-node probe of the black box is
   built -- a fraction of a second -- and every design variable is confirmed to be an
   independent variable the box accepts, and the objective and every constraint to be a quantity
   it publishes. A misspelled design variable is a message naming it with suggestions, not an
   OpenMDAO error thrown out of ``setup`` after a minute of building.
2. **Declare, then build.** Design variables, constraints and the objective are registered on
   the box's own group *before* ``setup``, because OpenMDAO requires it. Registering a design
   variable changes no OpenConcept behaviour; it states which of the box's existing independent
   variables a driver may move.
3. **Converge a baseline.** The continuation ladder is walked once. The driver's first function
   evaluation must begin from a converged aircraft.
4. **Drive.** Every later design starts from its predecessor's converged state.
5. **Evaluate the basis at the optimum** and report both the design change and the traceability
   matrix.

Design variables
----------------

.. code-block:: yaml

   design_variables:
     - {name: ac|geom|wing|S_ref, lower: 90.0, upper: 180.0, units: m**2}
     - {name: ac|geom|wing|AR, lower: 7.0, upper: 13.0}
     - {name: ac|geom|wing|c4sweep, lower: 15.0, upper: 32.0, units: deg}
     - {name: ac|geom|wing|taper, lower: 0.12, upper: 0.35}
     - {name: ac|propulsion|engine|rating, lower: 18.0e3, upper: 34.0e3, units: lbf}

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

**Constraints** are scaled by their own limit, automatically. Without it a climb gradient in
hundredths of a radian is numerically invisible next to a field length in thousands of feet.

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
requirement is satisfied. Both halves matter: a driver that stops on its iteration limit inside
an infeasible region has not solved the problem, and a report that calls that an optimum is
wrong. ``cdadt optimize`` exits non-zero in that case, so a shell script cannot archive a
failure as a success.

The shipped study
-----------------

Minimizing fuel with reserves over the wing planform and the engine rating, subject to 14 CFR
25.113, 25.121(b)(1)(i) and the engine deck's throttle limits in climb and cruise:

=============================  ===========  ===========  ==========
Quantity                       Baseline     Optimum      Change
=============================  ===========  ===========  ==========
Fuel with reserves (kg)        18,596.8     15,991.4     **-14.0%**
Block fuel (kg)                15,976.6     13,751.3     -13.9%
Maximum takeoff weight (kg)    78,345.0     71,959.3     -8.2%
Operating empty weight (kg)    41,748.2     37,968.0     -9.1%
Engine rating (lbf)            27,000       21,911       -18.9%
Balanced field length (ft)     5,247.7      6,263.7      +19.4%
=============================  ===========  ===========  ==========

Four of four requirements met, one active. The active one is the **climb throttle limit**, and
that is the interesting result: the design is not limited by the runway or by the second-segment
climb gradient, both of which retain large margins, but by the engine deck running out of
throttle in the climb. The field length grows by 19% and remains comfortably inside the 8000 ft
limit -- the optimizer spends the margin it has and stops where it runs out of thrust.

Reproduce it with:

.. code-block:: bash

   cdadt optimize cases/b738_optimization.yaml

Read :doc:`validation` before quoting these numbers. In particular the engine-out gradient is
evaluated clean rather than in the takeoff configuration §25.121(b) specifies, and the engine is
a scaled deck rather than a redesigned engine -- an 18.9% reduction in rating is a long way down
the surrogate.
