Verification
============

Verification asks whether the equations are being solved right. :doc:`validation` asks whether
they are the right equations. This page is the first question, answered with measurements rather
than assurances -- every number below is produced by a test in the suite, not transcribed from a
previous session.

Five studies: discretization, solver, derivatives, reproducibility, and optimality.

.. contents::
   :local:
   :depth: 1

Discretization: is the answer converged in the grid?
-----------------------------------------------------

The black box integrates fuel burn with Simpson's rule over ``num_nodes`` points per phase and
solves each phase duration against a target on that same grid. Every result therefore carries a
discretization error, and a number quoted without knowing that error is a number quoted without
knowing how many of its digits mean anything.

Run at a uniform solver tolerance of 1e-9 on every grid -- the tolerance the shipped cases
request, which every grid reaches -- so that only discretization varies. Comparing grids
converged to different residuals is the most common way a grid study reaches a wrong
conclusion.

.. list-table:: Grid refinement, shipped B738 sizing case
   :header-rows: 1
   :widths: 10 22 22 22 24

   * - ``num_nodes``
     - MTOW (kg)
     - Fuel with reserves (kg)
     - Balanced field (ft)
     - Relative error in fuel
   * - 11
     - 78345.0201
     - 18596.8267
     - 5247.7076
     - 2.54e-05
   * - **21** (shipped)
     - **78345.6435**
     - **18597.3177**
     - **5247.7948**
     - **9.84e-07**
   * - 31
     - 78345.6217
     - 18597.3005
     - 5247.7919
     - 5.76e-08
   * - 41
     - 78345.6211
     - 18597.3001
     - 5247.7918
     - 3.56e-08
   * - 61
     - 78345.6202
     - 18597.2994
     - 5247.7917
     - 3.64e-09
   * - 81
     - 78345.6203
     - 18597.2994
     - 5247.7917
     - reference

**Observed order of convergence.** From the 21/41/81 triple, which has a constant refinement
ratio of two,

.. math::

   p = \frac{\ln\bigl((f_{21} - f_{41}) / (f_{41} - f_{81})\bigr)}{\ln 2} \approx 4.7

Simpson's rule is fourth order. Measuring about that is what says the integration is behaving as
designed rather than agreeing by accident; measuring first order would mean something in the
chain had quietly become piecewise constant.

**Conclusions.** The shipped 21-node grid is converged to about 1e-6 relative, which licenses
the four decimal places quoted elsewhere in this documentation. The 11-node grid used for
optimization carries 2.5e-5 error in fuel -- four orders of magnitude smaller than the 14%
design change that study resolves, so it is a sound trade.

Tested by :mod:`tests.test_verification_grid`.

Solver: is the coupled system actually converged?
--------------------------------------------------

**A failed solve raises.** With one Newton iteration allowed, the design mission cannot
converge and the run raises rather than returning numbers. This is the most important negative
test in the repository, because every other check assumes that a result which came back is a
result that converged. Setting ``err_on_non_converge: false`` returns the state reached instead;
supported, because a study may want to inspect a failure, and required to be explicit, because
that state is indistinguishable from a converged one in a results table. The test that exercises
it confirms the balanced field is *not* balanced in that state -- which is exactly how a
non-converged run betrays itself.

**The shipped tolerance has three decades of margin, and this was re-measured after the
case-file rewrite.** Every tolerance probed down to 1e-12 is reachable at 11 nodes. That is a
change: an earlier case-file layout stalled at about 9e-9 on the 31- and 41-node grids, and this
page previously reported that stall as an absolute floor of the box. It was not. The difference
is that the ground-roll true-airspeed seeds are now ordinary initial conditions, re-applied
before every continuation rung and before the design run, where the earlier layout wrote them
once at the start. The shipped 1e-9 therefore sits well clear of anything, which is why the grid
study above can refine at the tolerance the results are actually produced at rather than one
decade looser.

**The answer does not depend on how tightly it was converged.** Between 1e-7 and 1e-9 the
residual falls by two orders of magnitude and no reported quantity moves by more than 1e-7
relative. Had the results moved with the residual, the digits would have been solver artefacts.

**The balanced field solve is a solve.** The box seeds the decision speed at 20 m/s and solves
it implicitly; the converged V\ :sub:`1` of 135.1 kn is more than three times that seed, and the
continue and abort distances match to 1e-6. A run that reported a plausible field length while
V\ :sub:`1` sat near its seed would have passed every other check here.

Tested by :mod:`tests.test_verification_solver`.

Derivatives: are the gradients the optimizer steps on correct?
---------------------------------------------------------------

Every design change cdadt reports comes from an optimizer following total derivatives taken
across the whole Newton-converged coupled system. If those are wrong the optimizer still
converges -- to the wrong design, confidently, with nothing in the results table to show it.

Checked against finite differences of the converged model at three step sizes:

.. list-table:: Total derivative agreement, worst case over 3 responses x 3 design variables
   :header-rows: 1
   :widths: 30 30 40

   * - Relative FD step
     - Worst relative error
     - Limited by
   * - 1e-5
     - 4.92e-04
     - truncation
   * - **1e-6**
     - **1.10e-04**
     - **minimum**
   * - 1e-7
     - 9.13e-04
     - round-off

Three step sizes rather than one, because a single one cannot distinguish a correct derivative
from a coincidence. A *wrong* analytic derivative differs from finite differences by a roughly
constant amount at every step, giving a flat error curve. A correct one is limited by truncation
at large steps and round-off at small ones, so its error has a minimum in between. The minimum
is the evidence; the number at the bottom of it is only the headline.

The floor on the agreement is set by the solver: the converged state is good to about 1e-9, so
differencing it over a relative step of 1e-6 cannot do better than roughly 1e-4. Agreement at
that level is the correct expectation, and demanding 1e-10 would be demanding the finite
difference be more accurate than the thing it differences.

**Signs are checked separately**, because a sign error is the failure that sends an optimizer
confidently the wrong way: raising aspect ratio must cut fuel and improve the engine-out climb
gradient, and adding thrust must shorten the field. All three hold.

Tested by :mod:`tests.test_verification_derivatives`.

Reproducibility: is the answer a property of the case, or of the route to it?
------------------------------------------------------------------------------

The box is converged by walking a continuation ladder, because a Newton solver started cold on a
2800 nmi mission at FL350 does not reach it. That raises a question which must be answered before
any number means anything: **is the ladder a path to the solution, or part of it?**

Three genuinely different ladders were run -- the shipped two-rung ladder, a four-rung ladder
that steps range and altitude together, and a three-rung ladder that goes to cruise altitude
first and stretches range afterwards. Every route that reaches the design mission reaches the
**same aircraft to better than 1e-7 relative**, on all of MTOW, OEW, MLW, both fuel figures,
field length, V\ :sub:`1` and V\ :sub:`2`.

Not every route arrives. A ladder that stalls is a statement about that ladder, and the test
distinguishes the two cases explicitly rather than counting a failure to arrive as evidence of
path independence: routes that do not converge are recorded, and at least two must arrive before
the comparison is allowed to claim anything.

**Determinism** is checked separately -- the same case run twice gives bit-identical results, so
nothing in the chain depends on iteration order, hashing or wall clock.

Tested by :mod:`tests.test_verification_reproducibility`.

Optimality: is the reported optimum actually an optimum?
----------------------------------------------------------

A driver reporting success means it satisfied its own termination criteria. A run that stalled on
its iteration limit prints exactly like a converged optimum.

So the optimum is probed directly. Each design variable is stepped 2% in both directions, the box
is reconverged at the perturbed design, and the objective and the full certification basis are
re-evaluated. Around a genuine constrained minimum every feasible direction must either fail to
improve the objective or leave the feasible set, and that is what is observed.

**The active set is checked for honesty** as well: a constraint the report labels ACTIVE must
really sit on its bound, and one labelled MET must not. The traceability matrix is the artefact a
certification argument is built from, and a mislabelled row would be wrong in the one place a
reader looks.

Tested by :mod:`tests.test_verification_optimality`.

Internal consistency
--------------------

Every result is read out of the box independently, by the discipline that owns it, from its own
path and in its own units. Nothing forces them to agree, so whether they do is a real check --
and the one that catches the two errors most likely to survive everything else: a response wired
to the wrong path, and a response declared in the wrong units.

A mass read in pounds and reported as kilograms is off by 2.2. It would still be positive, still
vary sensibly with the design, still reproduce run to run, and still print to four decimals. What
it would not do is add up.

Checked: the weight closure in both kilograms and fractions; the ordering OEW < MLW < MTOW;
structure as a fraction of empty weight; installed engine mass against single-engine mass times
engine count; V\ :sub:`2` > V\ :sub:`1`; the balanced field identity; fuel accumulating
monotonically; flaps raising CL\ :sub:`max`; tail areas as a credible fraction of the wing; every
phase duration positive; every throttle history inside its physical bounds. Units are separately
round-tripped through a second unit on seven representative responses, which is what proves the
declared units are *applied* on the way out rather than merely recorded.

Tested by :mod:`tests.test_verification_consistency`.

The environment reproduces from the file that describes it
------------------------------------------------------------

An environment that works is not the same as a recipe that rebuilds it, and the difference is
only visible if someone rebuilds. So the environment was built from scratch, on a clean name,
from ``environment.yml`` alone, and the whole suite was run in it.

It failed twice before it passed, and both failures were defects in the recipe rather than in
the code:

*The editable installs were recorded as PyPI pins.* ``conda env export`` had written
``cdadt==0.2.0`` and ``openconcept==1.2.6`` into the pip section. A build from that file would
have fetched a cdadt that does not exist on PyPI and a *different* OpenConcept than the local
clone the results depend on.

*The BLAS variant selector had been stripped.* ``--no-builds`` reduces ``libblas`` to a bare
version, and conda then resolved the MKL-backed build. The environment built cleanly, imported
NumPy cleanly, and aborted the interpreter with ``0xc06d007f`` inside ``numpy.linalg.solve`` the
moment OpenConcept was imported -- exactly the failure :doc:`install` warns about, reintroduced
by the file meant to prevent it. Note the failure mode: the interpreter dies, so ``pytest``
terminates mid-collection with a fatal exception instead of reporting a failed test.

With both fixed, a from-scratch environment ran the whole suite green and reproduced every
documented number to all printed digits -- MTOW 78345.6435 kg, fuel with reserves 18597.3177 kg,
balanced field length 5247.7948 ft. The verification environment was then removed; the recipe,
not the environment, is the artefact.

What the suite establishes, by claim class
-------------------------------------------

Every test declares which kind of claim it makes, so that "the suite passes" can be read as a
statement about what has actually been established:

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Marker
     - Count
     - Claim
   * - ``unit``
     - 159
     - One cdadt class behaves as specified, with no model built
   * - ``contract``
     - 14
     - The cdadt/OpenConcept boundary and the ownership map hold
   * - ``integration``
     - 41
     - A real OpenConcept model builds, converges and is driven
   * - ``verification``
     - 39
     - The equations are solved right: grid, solver, derivatives, reproducibility, optimality, consistency
   * - ``validation``
     - 19
     - The right equations were solved: against the reference example, and against physical reality
   * - **total**
     - **272**
     - ~8 minutes; ``-m "not slow"`` runs 216 of them in under two

Coverage
--------

**100% of statements and 100% of branches**, enforced rather than reported: ``fail_under = 100``
in ``pyproject.toml`` means a run with ``--cov`` fails if a single line or branch of cdadt goes
unexercised.

.. code-block:: bash

   pytest -q --cov=cdadt

That is a defensible target here specifically because cdadt computes no physics. There is no
solver to drive into an exotic state and no correlation valid only over some range -- only
interface, validation, routing and reporting, all of which are reachable from a test. A line
that cannot be reached is therefore either dead code or a missing test, and both are worth
finding.

There is deliberately **no** ``exclude_lines`` list beyond coverage's default
``# pragma: no cover``, which nothing in the package currently uses. An exclusion list is how a
coverage number quietly stops meaning anything.

Turning branch coverage on was itself worth it: statement coverage reached 100% while four
branches were still unexercised -- three ``--json``-less paths through the command line and the
non-pyOptSparse path through the driver settings. Each is now covered by a test that says what
it is checking.

.. code-block:: bash

   pytest -q                      # everything
   pytest -q -m verification      # this page
   pytest -q -m validation        # the next page
   pytest -q -m "not slow"        # the fast loop
