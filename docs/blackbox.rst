The OpenConcept black box
=========================

The sizing analysis cdadt drives is an OpenConcept group, named in the case file:

.. code-block:: yaml

   black_box:
     model: openconcept.examples.B738_sizing:B738SizingMissionAnalysis
     num_nodes: 21

cdadt sets its inputs, converges it, and reads its outputs. It does nothing else to it.

What "black box" means here, precisely
--------------------------------------

Three specific things, each enforced by :mod:`tests.test_boundary` rather than claimed here.

**No cdadt module imports OpenConcept.** Every cdadt source file is parsed and its imports
inspected. The dependency is a string in a YAML file, resolved by :func:`importlib.import_module`
at run time. That is a stronger guarantee than "cdadt does not modify OpenConcept", because it
removes the possibility rather than the practice: there is no imported class to subclass, no
imported component to re-wire, and no way for "compose one part of it" to drift into
"reimplement half of it".

**No cdadt class inherits from OpenConcept.** Overriding a method changes a component's
behaviour while leaving its source untouched -- a modification no diff of the clone would ever
show. Every class cdadt defines is walked and its whole inheritance chain checked.

**No OpenConcept source is modified.** ``git status`` is run in the OpenConcept working tree and
any uncommitted change fails the suite. Separately, every commit the installed clone carries
that upstream does not is examined, and the files it touches are compared against the modules
actually loaded when a box is built -- taken from :data:`sys.modules`, so it is measured rather
than assumed. The clone these results were produced against carries none: it is exactly
``mdolab/openconcept`` ``origin/main``. See :doc:`validation`.

What is inside
--------------

Everything that computes anything. For the shipped case,
``openconcept.examples.B738_sizing:B738SizingMissionAnalysis`` contains:

- the per-phase aircraft model OpenConcept instantiates once per mission phase;
- the component-by-component parasite drag buildup and the parabolic drag polar;
- the rubberized CFM56 engine deck -- thrust and fuel flow scaled from a fixed deck;
- the jet-transport empty-weight correlations, structure and equipment;
- horizontal and vertical tail sizing by tail volume coefficient;
- the clean and takeoff maximum lift estimates;
- the weight closure that makes takeoff weight consistent with the fuel burned;
- ``FullMissionWithReserve``: the balanced-field takeoff, climb, cruise, descent, the Part 25
  reserve diversion and loiter, and the implicit solves that make all of it consistent.

cdadt computes none of it.

What is fixed inside, and cannot be reached
-------------------------------------------

Some of the box's modelling choices are *options* of its components rather than inputs to them.
They are therefore not settable from a case file, and no amount of configuration will change
them. They are listed because a design study that does not know about them is quoting numbers
whose assumptions it cannot state:

===============================================  ===========================================
Horizontal tail volume coefficient, 1.00         Raymer Table 6.4, jet transport
Vertical tail volume coefficient, 0.09           Raymer Table 6.4, jet transport
Tail lever arm, half the fuselage length         An estimate made inside the box
Maximum landing weight, 0.8 x MTOW               An estimate, used only to size the gear
Ultimate load factor and structural allowances   Constants of the weight correlations
Engine deck, CFM56, scaled by rated thrust       Not a cycle analysis; a scaled surrogate
Fuel burn integration, Simpson's rule            Why ``num_nodes`` must be odd
===============================================  ===========================================

Changing any of these means changing the box, which means writing or choosing a different
sizing analysis and naming it in ``black_box.model``. It does not mean editing OpenConcept.

What goes in
------------

The complete, generated list is :doc:`interface`; ``cdadt inspect <case>`` prints it. In
summary, cdadt sets:

Design parameters
    Every ``ac|`` variable the box publishes as an independent variable -- 34 of them for the
    shipped case: the wing planform, the empennage shape, the fuselage, nacelles and gear, the
    aerodynamic parameters, the engine rating and count, the payload and the cabin. Declared in
    the ``aircraft`` section and routed to the discipline that owns each.

The mission
    ``mission_range``, ``reserve_range``, ``cruise|h0``, ``reserve|h0``, ``loiter|h0``,
    ``loiter_duration`` and ``takeoff|h``, plus the equivalent-airspeed and vertical-speed
    schedule flown at every node of every steady-flight phase, and a true-airspeed seed for the
    three ground-roll phases. See :doc:`mission`.

Starting guesses
    Values written onto the box's coupled states before the first solve. Maximum takeoff weight
    is the important one. These are held to a looser standard than design parameters on
    purpose: they are *outputs* the box solves for, so they are not independent variables and
    can never be design variables, but where Newton starts still decides whether it arrives.

Solver settings
    Iteration limit and tolerances for the Newton solve that closes the weight loop and the
    mission's balances together, plus whether a failed solve raises.

Design variables, an objective and constraints
    For an optimization, declared on the box's group before ``setup`` using OpenMDAO's public
    API. Declaring a design variable does not change what any OpenConcept component computes;
    it states which of the box's *existing* independent variables a driver may move.

What comes out
--------------

Every discipline reads its own responses. The ones that carry the certification argument:

===============================  ===================================================  =======
Response                         Path inside the box                                  Units
===============================  ===================================================  =======
``block_fuel``                   ``mission.descent.fuel_burn_integ.fuel_burn_final``  kg
``total_fuel``                   ``mission.loiter.fuel_burn_integ.fuel_burn_final``   kg
``takeoff_field_length``         ``mission.bfl.distance_continue``                    ft
``abort_distance``               ``mission.bfl.distance_abort``                       ft
``V1``                           ``mission.takeoff|v1``                               kn
``V2``                           ``mission.engineoutclimb.takeoff|v2``                kn
``engine_out_climb_gradient``    ``mission.engineoutclimb.gamma``                     rad
``MTOW``                         ``ac|weights|MTOW``                                  kg
``OEW``                          ``ac|weights|OEW``                                   kg
===============================  ===================================================  =======

Two of those deserve a note.

``total_fuel`` is read at the end of *loiter*, not at the end of descent. Fuel accumulates
across every phase because the box links each phase's final integrated state to the next
phase's initial value, and loiter is the last phase it accumulates through. Closing the weight
loop on block fuel instead would size an aircraft that carries no reserves -- and would converge
just as readily.

``takeoff_field_length`` and ``abort_distance`` are equal at convergence by construction: the
box solves V\ :sub:`1` implicitly so that continuing after an engine failure and rejecting the
takeoff cover the same distance. That is what makes it a *balanced* field length, and the
validation suite asserts the equality rather than trusting it.

The twelve phases
-----------------

**Takeoff, balanced field**
    ``v0v1`` ground roll to the decision speed; ``v1vr`` decision speed to rotation; ``rotate``
    the transition to 35 ft; ``v1v0`` the rejected takeoff; and ``EngineOutClimbAngle``, a
    single-point engine-out climb condition at V\ :sub:`2`.

**Design mission**
    ``climb``, ``cruise``, ``descent``.

**Part 25 reserves**
    ``reserve_climb``, ``reserve_cruise``, ``reserve_descent``, ``loiter``.

Why the boundary is where it is
-------------------------------

An earlier version of cdadt drew it differently: it imported twenty-odd OpenConcept components
and re-wired them into its own discipline groups, which reproduced the reference example
exactly and was, in every honest sense, a reimplementation of OpenConcept's own example living
inside cdadt. It was called a black box and was not one.

The boundary that survives is the one that cannot be crossed by accident. cdadt does not import
OpenConcept, so it cannot compose it, so the question of how much composition is too much never
arises. What cdadt contributes is on its own side of the line: the case file, the discipline
model of the interface, the continuation ladder, the certification basis, the optimizer and the
reporting.

The cost is real and is stated in :doc:`validation`: cdadt can only constrain what the box
publishes, so a regulation whose governing quantity the box does not produce cannot be enforced.
