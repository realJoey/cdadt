Validation
==========

What has been validated, against what -- and, with equal prominence, what has not.

The reference comparison
------------------------

cdadt's B738 sizing analysis is compared, quantity by quantity, against OpenConcept's own
``B738_sizing.py`` example. Both are run in the same process, in the same environment, on the
same aircraft and the same mission, at 21 nodes per phase.

============================  ==============  ==============  ==========
Quantity                      OpenConcept     cdadt           Difference
============================  ==============  ==============  ==========
MTOW (kg)                     78345.6435      78345.6435      0
OEW (kg)                      41748.3258      41748.3258      0
MLW (kg)                      62676.5148      62676.5148      0
Block fuel (kg)               15977.0628      15977.0628      0
Total fuel (kg)               18597.3177      18597.3177      0
Balanced field length (ft)    5247.7948       5247.7948       0
Abort distance (ft)           5247.7948       5247.7948       0
CL\ :sub:`max` cruise         1.427435        1.427435        0
CL\ :sub:`max` takeoff        2.267305        2.267305        0
Wing MAC (m)                  4.268446        4.268446        0
Horizontal tail area (m²)     27.933210       27.933210       0
Vertical tail area (m²)       20.210103       20.210103       0
============================  ==============  ==============  ==========

"0" means agreement to better than 10\ :sup:`-6` relative, which is the Newton solver's own
convergence rather than an engineering tolerance. The two models are solving the same
equations; anything looser would be hiding a real difference.

The reference is **run, not quoted**. ``tests/test_validation_b738.py`` imports
``run_738_sizing_analysis`` and compares live. A table of numbers pasted from a previous
session only tests that nobody edited the table.

The one deliberate difference
-----------------------------

:class:`~cdadt.model.Part25PhaseModel` -- used by the optimization example, and **not** by the
validation above -- evaluates the engine-out climb condition with takeoff flaps deployed,
because 14 CFR 25.121(b) specifies them. OpenConcept's example evaluates it clean.

====================================  ==============  ==========
Model                                 Climb angle     Gradient
====================================  ==============  ==========
``JetTransportPhaseModel`` (clean)    0.0579 rad      5.8%
``Part25PhaseModel`` (takeoff flaps)  0.0422 rad      4.2%
====================================  ==============  ==========

Both are above the 2.4% §25.121(b)(1)(i) minimum. The flapped value is the one to quote against
the regulation. This is the only place cdadt intentionally disagrees with the reference, and
it is a subclass rather than a flag so that the choice is visible in the model a run was made
with.

What the tests establish
------------------------

Every test declares the class of claim it makes, so that "the suite passes" can be read as a
statement about what has been established.

``unit``
    One cdadt class in isolation, no OpenMDAO model. Schedule resampling, definition parsing,
    requirement margins, constraint scaling and status logic.

``integration``
    A real OpenMDAO model, built and run. Each discipline against the formula it implements --
    the trapezoidal MAC, Raymer's tail volume relations, fuel-burn integration over a known
    duration, the halving of thrust and fuel flow with one engine out, the drag increase with
    flaps deployed -- plus the assembled models' promotion structure.

``contract``
    The OpenConcept boundary itself: the clone is clean, and no cdadt class inherits from an
    OpenConcept class.

``validation``
    cdadt against an external reference that is run, not quoted. The table above, plus three
    internal consistency conditions that would each catch a specific wrong answer: the balanced
    field is actually balanced, the weight loop is actually closed, and total fuel exceeds
    block fuel so the aircraft is carrying its reserves.

Run them:

.. code-block:: bash

   pytest -q -m "not slow"    # 43 tests, about a second
   pytest -q                  # everything, about 13 seconds

What has **not** been validated
-------------------------------

This section is the honest half and is meant to be read.

**Nothing has been validated against flight test or a published type certificate.** The
comparison above establishes that cdadt assembles OpenConcept's components the way OpenConcept
does. It says nothing about whether OpenConcept's empirical drag buildup, weight correlations
or engine surrogates are right for any real aeroplane. Every claim inherits OpenConcept's
accuracy and adds nothing to it.

**The optimum has not been validated at all.** ``examples/optimize_b738.py`` produces a
converged, feasible design. Whether a 13-aspect-ratio wing at 100 m² is buildable is a question
the empirical weight correlations underneath are not qualified to answer, and the aspect ratio
sitting on its upper bound says so plainly.

**The approach-speed analysis is unvalidated physics**, in the sense that no reference case has
been checked against it. Its components are OpenConcept's and its formula is the regulation's,
and it is tested against a hand calculation -- but a hand calculation confirms arithmetic, not
that the model is the right model.

**Node-count convergence has not been studied.** Results are reported at 21 nodes per phase
because that is what the reference example uses. The example optimization runs at 11 for speed;
on the B738 baseline that changes total fuel from 18597.3177 kg to 18596.8267 kg, a difference
of 0.0026%. That is evidence about one aircraft on one mission, not about the discretization.

**Derivative accuracy has not been checked** against complex step or finite difference. The
optimizer converges and the constraints are satisfied, which is weaker evidence than a
``check_totals`` and should not be mistaken for it.

**No requirement outside the four in** :doc:`certification` **is modeled.** A design that
satisfies this basis is not a certifiable aeroplane. It is an aeroplane that satisfies four
requirements.
