The mission
===========

The mission is what the aircraft is sized against, and it is owned by the
:class:`~cdadt.disciplines.performance.Performance` discipline as a
:class:`~cdadt.mission.MissionProfile`. It has three parts: the mission-level parameters, the
per-phase schedules, and the continuation ladder.

What is flown
-------------

The shipped case flies a 2800 nmi design mission at FL350, followed by a 200 nmi diversion at
FL150 and a 30-minute loiter:

.. code-block:: yaml

   mission:
     parameters:
       mission_range: {value: 2800, units: nmi}
       cruise|h0:     {value: 35000, units: ft}
       reserve_range: {value: 200, units: nmi}
       reserve|h0:    {value: 15000, units: ft}

``loiter|h0``, ``loiter_duration`` and ``takeoff|h`` are not set, so they keep the box's own
defaults: 1500 ft, 30 minutes and sea level. That is a deliberate choice about which numbers the
case owns; setting them is a one-line addition.

The schedules
-------------

Each of the seven steady-flight phases is flown along an equivalent-airspeed and vertical-speed
schedule. All seven must be present. There is no default profile to fall back on: an
unscheduled phase flies whatever placeholder its component declared, which is a different
mission than the one the case asks for, quietly.

.. code-block:: yaml

     schedule:
       climb:
         Ueas: {value: [230, 252], units: kn}
         vs:   {value: [2300, 400], units: ft/min}
       cruise:
         Ueas: {value: [252, 252], units: kn}
         vs:   {value: 0, units: ft/min}
       descent:
         Ueas: {value: [252, 250], units: kn}
         vs:   {value: [-1300, -800], units: ft/min}

A value may be one number, two numbers to interpolate between across the phase, or exactly
``num_nodes`` numbers. Any other length is an error. Both ``Ueas`` and ``vs`` are required in
every entry, including inside a continuation step: setting one alone flies a profile that is
half inherited from whatever was set before.

The three ground-roll phases -- ``v0v1``, ``v1vr``, ``v1v0`` -- are not scheduled. They
integrate acceleration from a standstill, so what they need is a starting guess for true
airspeed rather than a profile, and ``takeoff_speed_guess`` supplies it once, before the first
solve.

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
       - description: short range at low altitude, gentle descent
         parameters:
           mission_range: {value: 500, units: nmi}
           cruise|h0:     {value: 5000, units: ft}
           reserve_range: {value: 100, units: nmi}
           reserve|h0:    {value: 1000, units: ft}
         schedule:
           descent:
             Ueas: {value: [252, 250], units: kn}
             vs:   {value: [-800, -800], units: ft/min}

       - description: design range and altitude, still with the gentle descent
         schedule:
           descent:
             Ueas: {value: [252, 250], units: kn}
             vs:   {value: [-800, -800], units: ft/min}

Each step is written on top of the design profile and converged, so the solver enters the next
step from a converged neighbour. The last step is followed by the design mission itself. The
shipped ladder is two rungs: first shrink the mission to 500 nmi at 5000 ft and soften the
descent, then restore the range and altitude but keep the gentle descent, then finally apply
the real descent schedule.

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
