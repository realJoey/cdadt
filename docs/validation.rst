Validation
==========

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

**The boundary holds.** :mod:`tests.test_boundary` proves, rather than asserts, that no cdadt
module imports OpenConcept, that no cdadt class inherits from it, that the OpenConcept working
tree carries no uncommitted change, and that every settable variable of the built box is owned
by exactly one discipline. See :doc:`blackbox`.

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

**The numbers correspond to a specific OpenConcept checkout.** The installed clone is at commit
``5614b60``, which is ``origin/main`` plus three local commits. Those three touch only
``openconcept/aerodynamics/openaerostruct/aerostructural.py`` -- OpenAeroStruct 2.x and NumPy
compatibility fixes made in earlier, unrelated work -- and cdadt never loads that module. A
contract test verifies exactly that, by intersecting the files those commits touch with the
modules actually present in :data:`sys.modules` after a box is built. Nothing has ever been
pushed from that clone; ``origin`` is the upstream MDO Lab repository.

The clone was brought up to ``origin/main`` on 2026-07-28, from a checkout that was ten commits
behind. The only upstream change affecting a module cdadt loads was "Modify BFL residual (#86)"
in ``openconcept/mission/phases.py``, which refactors the decision-speed residual into a helper
that selects the same branch at convergence. **Every quantity on this page is unchanged to all
printed digits across that update**, balanced field length included, and the full suite passed
before and after. That is worth recording precisely because it could have gone the other way:
which OpenConcept is installed is part of the result, not part of the setup.

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

   pytest -q                    # 128 tests: unit, contract, integration, validation
   pytest -q -m validation      # the live comparison against OpenConcept's own example
   pytest -q -m contract        # the boundary
   pytest -q -m "not slow"      # the fast loop

Every test declares which class of claim it makes -- ``unit``, ``contract``, ``integration``,
``validation`` -- so that "the suite passes" can be read as a statement about what has actually
been established rather than as a single undifferentiated green tick.
