The mission
===========

The mission is what the aircraft is sized against. It is owned by the
:class:`~cdadt.disciplines.performance.Performance` discipline, as the same pair OpenConcept's
own run scripts write by hand: the :class:`~cdadt.mission.InitialConditions` -- everything
``set_values(prob, num_nodes)`` writes into the problem -- and the
:class:`~cdadt.mission.ContinuationLadder` that makes the hard ones reachable.

What is flown
-------------

The shipped case flies a 2800 nmi design mission at FL350, followed by a 200 nmi diversion at
FL150 and a 30-minute loiter:

.. code-block:: yaml

   initial_conditions:
     mission_range: {value: 2800, units: nmi}
     cruise|h0:     {value: 35000, units: ft}
     reserve_range: {value: 200, units: nmi}
     reserve|h0:    {value: 15000, units: ft}

``loiter|h0``, ``loiter_duration`` and ``takeoff|h`` are not set, so they keep the box's own
defaults: 1500 ft, 30 minutes and sea level. That is a deliberate choice about which numbers the
case owns; setting them is a one-line addition.

Names are written as the box publishes them, and resolved first as written and then under the
case's ``mission_path`` -- so ``cruise|h0`` and ``mission.cruise|h0`` both work, and a name that
resolves neither way is an error naming both attempts rather than a silently ignored line.

The schedules
-------------

Each steady-flight phase is flown along an equivalent-airspeed and vertical-speed schedule,
written under the phase's own name exactly as ``set_values`` writes it:

.. code-block:: yaml

   initial_conditions:
     climb.fltcond|Ueas:   {value: [230, 252], units: kn}
     climb.fltcond|vs:     {value: [2300, 400], units: ft/min}
     cruise.fltcond|Ueas:  {value: 252, units: kn}
     cruise.fltcond|vs:    {value: 0, units: ft/min}
     descent.fltcond|Ueas: {value: [252, 250], units: kn}
     descent.fltcond|vs:   {value: [-1300, -800], units: ft/min}

The shipped case schedules all seven steady phases -- climb, cruise, descent, the three reserve
phases and loiter -- and both quantities in each. Nothing forces that, and nothing can: to the
box these are ordinary variables carrying their own declared defaults. What an unscheduled phase
flies is whatever placeholder its component declared, which is a different mission than the one
the case appears to ask for. Schedule them all.

A value may be one number, two numbers to interpolate between across the phase (exactly as
``np.linspace`` does in OpenConcept's own run script), or exactly as many numbers as the
variable's shape. Any other length is an error rather than being broadcast, because broadcasting
would quietly fly a different mission.

The three ground-roll phases -- ``v0v1``, ``v1vr``, ``v1v0`` -- have no schedule. They integrate
acceleration from a standstill, so what they need is a starting guess for true airspeed:

.. code-block:: yaml

   initial_conditions:
     v0v1.fltcond|Utrue: {value: 100, units: kn}
     v1vr.fltcond|Utrue: {value: 100, units: kn}
     v1v0.fltcond|Utrue: {value: 100, units: kn}
     ac|weights|MTOW:    {value: 50.0e3, units: kg}

Those four are seeds, not designs. Maximum takeoff weight is an *output* -- it is what the weight
closure solves for -- and writing a value onto it before the first solve chooses where Newton
starts, not what it converges to. They are re-applied before every continuation rung and before
the design run, which is not cosmetic: it is why the box converges at every tolerance probed down
to 1e-12 on every grid, where writing them once left some grids stalling near 9e-9. See
:doc:`verification`.

Continuation, and why it is part of the interface
-------------------------------------------------

The black box is a coupled implicit system. Phase durations are solved against altitude and
range targets; throttle against zero acceleration; the decision speed V\ :sub:`1` so that the
continue and abort distances match; and above all of it, maximum takeoff weight against the fuel
burned. A Newton solver started cold on a 2800 nmi mission at FL350 does not generally reach it.

OpenConcept's own sizing example handles this by converging an easy mission first and stepping
up -- written as a sequence of bare assignments in a run script, between two ``run_model``
calls. cdadt does the same thing, as data:

.. code-block:: yaml

   continuation:
     - description: short low-altitude mission, gentle descent
       initial_conditions:
         mission_range:      {value: 500, units: nmi}
         cruise|h0:          {value: 5000, units: ft}
         reserve_range:      {value: 100, units: nmi}
         reserve|h0:         {value: 1000, units: ft}
         descent.fltcond|vs: {value: -800, units: ft/min}

     - description: design range and altitude, still with the gentle descent
       initial_conditions:
         descent.fltcond|vs: {value: -800, units: ft/min}

A rung takes ``description`` and ``initial_conditions``, nothing else. Each is written on top of
the case's own initial conditions -- which are re-applied underneath it every time -- and
converged, so the solver enters the next rung from a converged neighbour. The last rung is
followed by the design mission itself. The shipped ladder is two: first shrink the mission to
500 nmi at 5000 ft and soften the descent, then restore the range and altitude but keep the
gentle descent, and finally apply the real descent schedule.

Making this data rather than statements has three consequences worth the trouble. The ladder
travels with the mission it converges, so a case file that is copied and edited keeps working.
It can be inspected and changed per study without touching code. And it is the same object in a
sizing run and in the baseline of an optimization, so the two cannot silently diverge.

It also must not change the answer. The ladder is a path to the solution, not part of it, and
:mod:`tests.test_validation` is what proves the path taken here arrives where OpenConcept's own
run script arrives -- to 1e-6, with the reference executed live.

During an optimization
----------------------

The ladder is walked **once**, to establish the baseline. Every later design the driver proposes
starts from its predecessor's converged state, which is both far cheaper and far more reliable
than re-walking a ladder from a cold start at each iteration.

That is also why an optimizer will occasionally propose a design the solver cannot converge from
where it currently is. The shipped optimization case uses IPOPT for exactly this reason; see
:doc:`optimization`.

Reading the mission back
------------------------

:class:`~cdadt.disciplines.performance.Performance` reports what flying it produced: block fuel
and fuel with reserves, the balanced field length and its abort distance, V\ :sub:`1` and
V\ :sub:`2`, the engine-out climb gradient, the range actually flown on the design mission and
on the diversion, the throttle history of every phase, and every phase duration.

``mission_range_flown`` is worth checking on any run: the box solves the cruise duration to
reach the requested range, so a converged run flies exactly the mission that was asked for, and
a run where it does not has not converged.
