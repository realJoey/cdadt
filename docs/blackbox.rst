The OpenConcept black box
=========================

The mission analysis cdadt sizes against is
:class:`openconcept.mission.FullMissionWithReserve`. cdadt uses it as published: it sets its
inputs, converges it, and reads its outputs.

What "black box" means here, precisely
--------------------------------------

It means three specific things, each of which is enforced or is not claimed:

**No OpenConcept source is modified.** :mod:`tests.test_openconcept_boundary` runs
``git status`` in the OpenConcept working tree and fails the suite on any uncommitted change.

A clean working tree says nothing about local *commits*, and this development clone has three:
compatibility fixes to ``openconcept/aerodynamics/openaerostruct/aerostructural.py`` for
OpenAeroStruct 2.x and NumPy 2, made in earlier work. cdadt does not import that module. A
second test enforces exactly that -- it diffs the clone against ``origin/main``, imports the
whole of cdadt, and fails if any locally patched OpenConcept module turns out to be one cdadt
loads. The guarantee is therefore specific and true rather than broad and approximate.

**No OpenConcept class is subclassed.** Overriding a method would change a component's
behaviour while leaving its source untouched -- a modification no diff of the clone would
ever show. The same test walks every class cdadt defines and fails if any of them inherits
from anything in the ``openconcept`` namespace. cdadt *composes* OpenConcept components inside
its own :class:`openmdao.api.Group` subclasses instead.

**No OpenConcept physics is reimplemented.** Drag, thrust, weights, tail sizing, high lift,
atmosphere, the flight-condition solve, the balanced-field solve, the phase durations, the fuel
integration: all of it is OpenConcept's. cdadt contributes the assembly, the weight closure,
the continuation schedule, the certification basis and the optimizer.

What it does **not** mean: it is not a pure function. OpenConcept's documented interface
requires the caller to supply an aircraft model *class*, which OpenConcept instantiates once
per phase. That class is an input to the black box, and it is the input that is easiest to
mistake for an internal.

What goes in
------------

Design parameters
    Every ``ac|...`` variable in the :class:`~cdadt.aircraft.AircraftDefinition`, published as
    independent variables and promoted into the mission with ``promotes_inputs=["ac|*"]``. One
    value set at the top of the model reaches all twelve phases.

The mission profile
    ``mission_range``, ``reserve_range``, ``cruise|h0``, ``reserve|h0`` and the per-phase
    airspeed and vertical-speed schedules. Set on the mission group by
    :class:`~cdadt.mission.MissionProfile`. See :doc:`mission`.

The aircraft model class
    :class:`~cdadt.model.JetTransportPhaseModel`, or a subclass. OpenConcept requires it to
    consume the flight condition, the throttle, the lift coefficient, ``propulsor_active`` and
    the ``ac|`` parameters, and to produce ``thrust``, ``drag`` and ``weight``. Those three
    outputs are the entire contract in that direction.

Two closure values
    Maximum takeoff weight, and the maximum fuel load that sizes the fuel system. Both are
    fed back into the mission from the sizing loop above it, which is what makes the system
    coupled rather than feed-forward.

What comes out
--------------

:attr:`~cdadt.sizing.SizingAnalysis.RESULTS` is the full list, read on every run rather than
whichever few a caller asked for. The ones that carry the physics:

===============================  ==============================================  =======
Result                           OpenConcept path                                Units
===============================  ==============================================  =======
``block_fuel``                   ``mission.descent.fuel_burn_final``             kg
``total_fuel``                   ``mission.loiter.fuel_burn_final``              kg
``takeoff_field_length``         ``mission.bfl.distance_continue``               ft
``abort_distance``               ``mission.bfl.distance_abort``                  ft
``V1``                           ``mission.takeoff|v1``                          kn
``V2``                           ``mission.engineoutclimb.takeoff|v2``           kn
``engine_out_climb_angle``       ``mission.engineoutclimb.gamma``                rad
===============================  ==============================================  =======

``takeoff_field_length`` and ``abort_distance`` are equal at convergence by construction:
OpenConcept solves V\ :sub:`1` implicitly to make them so. A run where they differ has not
converged, and :mod:`tests.test_validation_b738` asserts it.

The twelve phases
-----------------

``FullMissionWithReserve`` builds, and cdadt's phase model is instantiated in, all of:

**Takeoff (balanced field)**
    ``v0v1`` ground roll to the decision speed; ``v1vr`` decision speed to rotation; ``rotate``
    the transition to 35 ft; ``v1v0`` the rejected takeoff; and ``EngineOutClimbAngle``, a
    single-point engine-out climb condition at V\ :sub:`2`.

**Design mission**
    ``climb``, ``cruise``, ``descent``.

**Part 25 reserves**
    ``reserve_climb``, ``reserve_cruise``, ``reserve_descent``, ``loiter``.

Fuel accumulates across all of them: OpenConcept's ``link_phases`` connects each phase's final
integrated state to the next phase's initial value. That is why total fuel is read at the end
of loiter and why sizing on block fuel would be wrong.

Why the boundary is a contract and not a wall
---------------------------------------------

A wall would mean cdadt could not use OpenConcept's components at all -- only the assembled
mission -- and would have to write its own drag buildup and weight correlations to fill the
aircraft model OpenConcept demands. That would be more code, less validated code, and a worse
answer. The boundary that matters is *modification*, not *import*: cdadt may use anything
OpenConcept publishes, exactly as published, and may change none of it. Both halves of that
sentence are tested.
