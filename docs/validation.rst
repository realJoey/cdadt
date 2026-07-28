Validation
==========

Validation asks whether the right equations were solved. :doc:`verification` asks whether they
were solved right; read that page too, since a validated model that is not converged is not
validated.

This page states what has been established, against what, and -- at least as prominently --
what has not. A tool whose results are quoted in a thesis needs both halves written down in the
same place.

What is established
-------------------

**cdadt reproduces OpenConcept's own B738 sizing example exactly.**

:mod:`tests.test_validation` imports ``openconcept.examples.B738_sizing.run_738_sizing_analysis``
and *runs it*, in the same process, in the same environment, at the same 21-node grid, and
compares every quantity cdadt reports against the reference problem it produced. The tolerance
is 1e-6 relative -- the Newton solver's own convergence, not an engineering tolerance.

The reference is run, not quoted. A table of numbers pasted from a previous session tests only
that nobody edited the table.

Twenty-two quantities are compared, including weights, geometry, maximum lift coefficients, the
structural weight breakdown, field length, decision and safety speeds, the climb gradient, both
fuel figures and the range flown; and separately, the full throttle history of climb, cruise and
descent node by node.

=================================  ================
Maximum takeoff weight             78,345.6435 kg
Operating empty weight             41,748.3258 kg
Maximum landing weight             62,676.5148 kg
Block fuel                         15,977.0628 kg
Fuel with reserves                 18,597.3177 kg
Balanced field length              5,247.7948 ft
Decision speed V1                  135.0566 kn
Takeoff safety speed V2            155.4331 kn
Horizontal tail area               27.9332 m²
Vertical tail area                 20.2101 m²
=================================  ================

Three further properties are checked on every validated run, because each of them can fail
while the numbers still look plausible:

- **The balanced field is balanced.** ``takeoff_field_length == abort_distance`` to 1e-6. The
  box solves V\ :sub:`1` to make it so; a run where they differ has not converged.
- **The weight loop is closed.** ``MTOW == OEW + payload + total_fuel`` to 1e-9.
- **Reserves are carried.** ``total_fuel > block_fuel``. Closing the loop on block fuel instead
  would size an aircraft with no diversion, and would converge just as readily.
- **The mission flown is the one requested.** ``mission_range_flown == 2800 nmi`` to 1e-6.

**The converged aircraft resembles a real 737-800.** The reference comparison above would pass
just as well if OpenConcept's physics were nonsense -- both sides would be wrong together. So
the design is separately checked against things known independently of the model:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - cdadt
     - Published / expected
     - Agreement
   * - Maximum takeoff weight
     - 78,346 kg
     - ~79,010 kg
     - −0.8%
   * - Operating empty weight
     - 41,748 kg
     - ~41,410 kg
     - +0.8%
   * - Wing span (consistency check)
     - 34.31 m
     - 34.32 m
     - −0.03%
   * - Cruise Mach
     - 0.785
     - 0.72–0.85
     - in band
   * - Cruise lift-to-drag ratio
     - 17.46
     - 14–20
     - in band
   * - Cruise TSFC (lb/lbf/hr)
     - 0.614
     - 0.50–0.75
     - in band
   * - Wing loading (kg/m²)
     - 628.8
     - 500–750
     - in band
   * - Sea-level thrust-to-weight
     - 0.313
     - 0.25–0.40
     - in band
   * - Empty weight fraction
     - 0.533
     - 0.45–0.60
     - in band

Two caveats, stated here rather than buried. The published figures are widely quoted 737-800
specification values, not certification data, and are checked to ±10% -- a plausibility check,
not a certification-grade validation. And **wing span is a consistency check, not validation**:
reference area and aspect ratio are *inputs* taken from the same source as the published span,
so agreement confirms the geometry is assembled and converted correctly and is not independent
evidence about the physics.

Cruise is also checked to be genuinely steady level flight -- thrust equals drag to 1e-6 -- which
is an identity rather than a band, and would fail if the trajectory being integrated were not the
one the mission claims to fly.

Tested by :mod:`tests.test_validation_physical`.

**The boundary holds.** :mod:`tests.test_boundary` proves, rather than asserts, that no cdadt
module imports OpenConcept, that no cdadt class inherits from it, that the OpenConcept working
tree carries no uncommitted change, and that every settable variable of the built box is owned
by exactly one discipline. See :doc:`blackbox`.

**The black box is the only one that could have been chosen.** ``B738SizingMissionAnalysis`` is
the sole analysis in OpenConcept's 30,000 lines that combines a balanced-field takeoff, Part 25
reserves, a closed weight loop and a scalable engine. That is a survey result, not a preference;
see :doc:`openconcept`.

**The optimization converges and is feasible.** :mod:`tests.test_optimization` runs the shipped
study end to end and asserts that the driver converged, that no requirement is violated, that
the objective actually improved, that the mission is still flown, and that the field length is
still balanced at the optimum.

What this does **not** establish
--------------------------------

**It says nothing about whether OpenConcept's model is right.** Agreement to 1e-6 means cdadt
drives the box correctly. The physics is OpenConcept's: empirical drag and weight buildups, a
scaled engine deck, tail volume coefficient sizing. Whether those are adequate for a given
design study is a separate question, and cdadt cannot answer it because cdadt does not compute
them.

**The numbers correspond to unmodified, upstream OpenConcept.** The installed clone is at
``0d2adeb``, exactly ``origin/main`` of ``mdolab/openconcept``: no local commits, no local
tags, and a clean working tree. Nothing has ever been pushed from it. The results on this page
can therefore be reproduced by anyone who clones OpenConcept and installs it, with no patching
step to describe and none to forget.

Two contract tests keep it that way rather than trusting it. One fails if the clone's working
tree is dirty. The other takes every commit the clone carries that upstream does not, and
intersects the files those commits touch with the modules actually present in
:data:`sys.modules` after a box has been built -- so a local commit affecting anything cdadt
loads fails the suite by name. Today the first set is empty and the test is vacuous, which is
the state it exists to protect.

Getting there was itself a result worth recording. The clone had been ten commits behind and
carried three local compatibility commits. Bringing it to ``origin/main`` changed exactly one
module cdadt loads -- ``openconcept/mission/phases.py``, "Modify BFL residual (#86)", which
refactors the decision-speed residual into a helper that selects the same branch at convergence
-- and dropping the three local commits changed none, since all three touched only
``aerodynamics/openaerostruct/aerostructural.py``. **Every quantity on this page is unchanged to
all printed digits across both changes**, balanced field length included, and the full suite
passed at each step. It could have gone the other way: which OpenConcept is installed is part of
the result, not part of the setup.

Known gaps in the certification argument
----------------------------------------

Every requirement cdadt can enforce is a function of a quantity the black box publishes. These
are the ones it cannot, and each is a real gap rather than an omission:

**§25.121(b) is evaluated clean.** The regulation specifies the second-segment gradient with the
gear retracted *and takeoff flaps set*. The box evaluates its engine-out climb condition in the
clean configuration, so the gradient is optimistic against the regulation as written. Correcting
it would mean changing what the box computes.

**Reference approach speed, §25.125 and the approach categories.** Needs the reference stall
speed at maximum landing weight in the landing configuration, which the box does not produce.
``ac|aero|Vstall_land`` is an input the case file sets, so a constraint derived from it would
constrain a constant.

**Minimum control speeds, §25.149.** No V\ :sub:`MCG` and no V\ :sub:`MCA`. The box has no
lateral-directional model, no rudder, and no engine-out yawing moment.

**Landing and approach climb, §25.119 and §25.121(c)/(d).** Not modelled.

**Landing field length.** The mission ends at the ground at the end of descent; there is no
landing roll.

**Centre of gravity and stability.** The :class:`~cdadt.disciplines.stability.Stability`
discipline is named for the domain tail volume coefficient sizing belongs to. It computes no
static margin, no centre-of-gravity range and no stability derivative, and cdadt makes no
stability claim of any kind.

**Fuel volume.** Fuel with reserves is a mass. Nothing checks that it fits in the wing.

Modelling limits worth stating with a result
--------------------------------------------

**The engine is a scaled deck, not a design.** Thrust and fuel flow are scaled from a fixed
CFM56 map by the rated thrust. A large change in rating extrapolates a surrogate rather than
redesigning an engine; the shipped optimization moves it by -18.9%, which is a long way down
that surrogate and should be read as a sizing trend rather than as an engine.

**The weight buildup is empirical and transport-shaped.** Roskam- and Raymer-class correlations
for a conventional metal jet transport. Design variable bounds should stay inside the range
those correlations were fitted over, which is why the shipped bounds are stated as an
engineering choice rather than a numerical one.

**Two of the box's own estimates enter the answer directly.** Maximum landing weight is taken
as 0.8 × MTOW, and the tail lever arm as half the fuselage length. Both are inside the box and
cannot be set from a case file. See :doc:`blackbox`.

How to re-establish all of this
-------------------------------

.. code-block:: bash

   pytest -q                    # 184 tests, ~3.5 minutes
   pytest -q -m validation      # this page: the live reference and the physical checks
   pytest -q -m verification    # grid, solver, derivatives, reproducibility, optimality
   pytest -q -m contract        # the boundary
   pytest -q -m "not slow"      # the fast loop

Every test declares which class of claim it makes -- ``unit``, ``contract``, ``integration``,
``verification``, ``validation`` -- so that "the suite passes" can be read as a statement about
what has actually been established rather than as a single undifferentiated green tick. The
counts per class are tabulated at the end of :doc:`verification`.
