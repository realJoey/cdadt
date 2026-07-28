.. _validation:

**********
Validation
**********

This page states what has been validated, against what, and -- with equal prominence -- what
has not. A validation page that lists only successes is a marketing document.

Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Part of the model
     - Status
     - Against what
   * - Wrapped OpenConcept components
     - **Validated**
     - Each provider vs. the bare component, identical inputs, agreement to 1e-12
   * - Aircraft-level assembly
     - **Validated**
     - vs. an assembly of the same components wired as ``B738_sizing.py`` wires them
   * - cdadt-authored landing components
     - **Validated**
     - Hand calculations; analytic partials vs. complex step
   * - Mission-level results
     - **Validated**
     - OpenConcept's published golden values, and a live run of its own example
   * - Engine-out climb gradient
     - **Deliberate divergence**
     - Differs from the reference by ~27%, for a stated reason — see below
   * - Optimization result
     - Regression only
     - Pinned against drift; the optimizer's own exit status is asserted

Mission results: validated against OpenConcept
==============================================

cdadt's providers wrap the very components OpenConcept's ``B738_sizing`` example uses, so a
cdadt sizing run and that example should produce the same aircraft. They do.

Two independent references are used, and the distinction matters.

**OpenConcept's published golden values.** Its own test suite
(``openconcept/examples/tests/test_example_aircraft.py``, ``B738SizingTestCase``) asserts
specific numbers at ``num_nodes=5``. Those are version-controlled truth from the dependency,
so checking against them does not depend on the reference example running here:

.. list-table::
   :header-rows: 1
   :widths: 26 26 26 22

   * - Quantity
     - OpenConcept golden
     - cdadt
     - Difference
   * - Block fuel
     - 35,213.767 lbm
     - 35,215.7 lbm
     - +0.006%
   * - Total fuel
     - 40,991.188 lbm
     - 40,991.9 lbm
     - +0.002%
   * - MTOW
     - 172,711.303 lbm
     - 172,711.4 lbm
     - +0.0002%

**A live run of the reference example**, compared quantity by quantity in the same session.
This covers ground the goldens do not: balanced field length, V₁, tail areas, maximum lift
coefficients, and both empty and landing weights. All agree to the tolerances in
``cdadt/tests/validation/test_against_openconcept.py``.

For a like-for-like comparison, these tests override ``mission|ground_roll_initial_speed``
back to OpenConcept's 2 m/s, so that every modeling choice is identical and only the
implementation differs.

A note on the environment
-------------------------

OpenConcept declares ``numpy >=1.20, <2``, and that bound is real. Under NumPy 2 its own
B738 test fails: the skin-friction correlation in its parasite-drag buildup evaluates
``log(0.06 Re)`` and ``sqrt(Re)`` at the first nodes of the ground roll and the solver
reaches ``NaN``. Under NumPy 1.26 the same test passes.

This is recorded because an earlier version of this page blamed the dependency for that
failure. It was an environment error: cdadt's environment had been built with ``--no-deps``,
which bypassed the bound. **The reference is the reference.** When it disagrees with cdadt,
or fails, the first suspect is cdadt's environment.

The engine-out climb gradient differs on purpose
================================================

One quantity does not match, and should not.

OpenConcept's ``B738AircraftModel`` selects the takeoff-configuration drag buildup for
``phase in ["v0v1", "v1v0", "v1vr", "rotate"]`` (``B738_sizing.py`` line 45).
``EngineOutClimbAngle`` is not in that list, so the reference evaluates the engine-out climb
with **clean** drag — no flaps.

14 CFR 25.121(b) specifies the second-segment climb "with the takeoff flaps, the landing gear
retracted". cdadt marks that phase as takeoff configuration and deploys the flaps, which adds
drag:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Engine-out climb gradient
     - Value
     - Configuration
   * - OpenConcept reference
     - 0.0579 rad
     - clean
   * - cdadt
     - 0.0422 rad
     - takeoff flaps

A §25.121 constraint evaluated on clean drag credits the design with climb performance it
does not have in the configuration the rule is written about. The gap is 27% — large enough
to change a design. cdadt therefore diverges deliberately, and the divergence is asserted by
a test rather than absorbed into a loosened tolerance.

What *is* validated
===================

Wrapped components
------------------

Each provider is run alongside the bare OpenConcept component it wraps, with identical
inputs, and the outputs must agree to a relative tolerance of ``1e-12``. The tolerance is
machine-level rather than engineering-level deliberately: a provider adds no physics, so
anything but an exact match means cdadt changed a number somewhere — a unit, a wiring, a
moment arm.

Covered: ``WingMACTrapezoidal``, ``CylinderSurfaceArea`` (fuselage and nacelle),
``HStabVolumeCoefficientSizing``, ``VStabVolumeCoefficientSizing``, ``CleanCLmax``,
``FlapCLmax``, ``ParasiteDragCoefficient_JetTransport`` in both clean and takeoff
configuration, ``PolarDrag``, and ``RubberizedTurbofan``.

Aircraft-level assembly
-----------------------

Component-level agreement does not prove the assembly is right: a wrong moment arm or a
dropped connection lives between the components, not inside them. So cdadt's aircraft-level
group is compared against an assembly built from the same OpenConcept components and wired
as ``openconcept/examples/B738_sizing.py`` wires them, both evaluated at the same fixed
takeoff weight so no solver is involved.

Compared, all agreeing to ``1e-12``: mean aerodynamic chord, fuselage and nacelle wetted
areas, both stabilizer areas, maximum landing weight, operating empty weight, and both
maximum lift coefficients.

The comparison is also checked for being non-vacuous — every compared quantity must be
finite, positive, and inside physically sensible bounds — so that two assemblies silently
producing their placeholder defaults could not pass it.

cdadt-authored components
-------------------------

The landing components are the only physics cdadt is responsible for rather than wrapping,
so they are held to a higher standard:

* **Hand calculations** worked out in each test docstring. For example, at
  :math:`V_\text{ref}` = 70 m/s with :math:`\bar{a}` = 3.5 m/s², the ground roll is
  4900/7 = 700 m; with a 300 m airborne segment the landing distance is 1000 m; factored by
  0.6 the field length required is 1666.67 m.
* **Analytic partials vs. complex step**, at a point where every input differs across the
  nodes, so a partial correct only for equal inputs would fail.
* **Direction and scaling checks** that a magnitude check alone would miss: the dispatch
  factor must *lengthen* the requirement, and doubling the approach speed must *quadruple*
  the ground roll.

Physical plausibility
=====================

Not validation, but worth recording. The sized B738 baseline against the real aircraft:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - cdadt
     - Boeing 737-800
     - Difference
   * - MTOW (kg)
     - 78,341
     - ~79,000
     - −0.8%
   * - OEW (kg)
     - 41,747
     - ~41,400
     - +0.8%

Both within 1%. This says the model is in the right regime; it is not evidence of
correctness, because the empirical buildups these come from were themselves calibrated on
aircraft of this class.

How to reproduce
================

.. code-block:: bash

   pytest -q -m validation -v          # the external comparisons
   pytest -q -m derivative -v          # partials vs. complex step
   pytest -q                           # everything, including the slow runs

What is still open
==================

Agreement with OpenConcept establishes that cdadt implements the same methods correctly. It
does not establish that those methods describe a real aircraft, because both codes share the
same empirical buildups. Two things would strengthen the claim:

1. **A second, independent reference** — published B737-800 payload-range or field-length
   data — validating the results as *performance* rather than as agreement with another
   code.
2. **Method-level validation** of the mission physics against hand calculations, the way the
   landing components are validated here.

The optimization result is pinned as a regression only. Its correctness rests on the sizing
model beneath it, which is validated, plus the optimizer's own reported exit status, which is
asserted.
