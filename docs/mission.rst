The mission and the continuation schedule
=========================================

A mission profile is two things: the mission to fly, and the sequence of easier missions that
gets the solver there.

The design mission
------------------

``cases/b738_mission.yaml``:

.. code-block:: yaml

   parameters:
     mission_range: {value: 2800, units: nmi}
     reserve_range: {value: 200, units: nmi}
     cruise|h0: {value: 35000, units: ft}
     reserve|h0: {value: 15000, units: ft}

   schedule:
     climb:
       Ueas: {value: [230, 252], units: kn}
       vs: {value: [2300, 400], units: ft/min}
     cruise:
       Ueas: {value: 252, units: kn}
       vs: {value: 0, units: ft/min}
     ...

Every steady-flight phase must be scheduled -- ``climb``, ``cruise``, ``descent``,
``reserve_climb``, ``reserve_cruise``, ``reserve_descent``, ``loiter``. There is no default to
fall back on, and an unscheduled phase would fly whatever placeholder its component declared,
converge, and report a number.

A schedule may be written three ways:

* **two values** -- endpoints, interpolated linearly across the phase's nodes;
* **one value** -- held constant;
* **``num_nodes`` values** -- used as given.

Anything else raises rather than broadcasting, because broadcasting a wrong-length schedule
flies a different mission than the one written down.

Why continuation is part of the interface
-----------------------------------------

OpenConcept's mission is a coupled implicit system. Phase durations are solved by
``BalanceComp`` against altitude and range targets; throttle is solved for zero horizontal
acceleration at every node; V\ :sub:`1` is solved so that the continue and abort distances
match; and cdadt's own weight closure is solved on top of all of it. A Newton solver started
on a 2800 nmi mission at 35,000 ft, from an aircraft that is only a guess, does not generally
converge.

The remedy is continuation: solve an easy mission, then a harder one starting from that
solution, and so on. OpenConcept's own example does exactly this, as a sequence of bare
``set_val`` and ``run_model`` calls in a run script.

cdadt makes it a declared object. The steps live in the case file, next to the mission they
converge, where they can be read, changed and tested:

.. code-block:: yaml

   continuation:
     - description: short range at low altitude, shallow descent
       parameters:
         mission_range: {value: 500, units: nmi}
         reserve_range: {value: 100, units: nmi}
         cruise|h0: {value: 5000, units: ft}
         reserve|h0: {value: 1000, units: ft}
       schedule:
         descent:
           Ueas: {value: [252, 250], units: kn}
           vs: {value: -800, units: ft/min}

     - description: design range and altitude, descent rate still shallow
       schedule:
         descent:
           Ueas: {value: [252, 250], units: kn}
           vs: {value: -800, units: ft/min}

Each step is applied **on top of the design profile**, so anything a step does not override is
the design value. That is why the second step needs only the descent schedule: its range and
altitudes are already the design ones.

A step must override ``Ueas`` and ``vs`` together or not at all. Overriding one alone leaves
the other at whatever the previous step set, which means flying a profile that appears nowhere
in the file; :class:`~cdadt.mission.MissionProfile` raises instead.

Ground-roll seeding
-------------------

The three ground-roll phases integrate acceleration from a standstill and their solver is
sensitive to where it starts, so their true-airspeed vectors are seeded --
``takeoff_speed_guess``, 100 kn by default.

The seeding happens **once**, before the first solve, not at each continuation step. Re-seeding
between steps would discard the converged state that each step exists to hand to the next.

Continuation changes nothing about the answer
---------------------------------------------

It changes whether the solver arrives. The design mission is applied and run one final time
after the last step, so the reported result is always a converged solve of the profile in
``parameters``, never a partially relaxed one. If the final solve fails, ``err_on_non_converge``
makes it raise: a non-converged mission still produces numbers, and they are indistinguishable
from results.

Reference
---------

.. automodule:: cdadt.mission
   :members:
   :noindex:
