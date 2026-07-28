.. _blackbox:

*********************
The mission black box
*********************

Full mission sizing -- balanced-field takeoff, climb, cruise, descent, Part 25 reserves, and
loiter -- is supplied by `OpenConcept <https://github.com/mdolab/openconcept>`_. cdadt does
not reimplement any of it and **never modifies OpenConcept**.

What "black box" means here
===========================

It means cdadt supplies inputs and reads named outputs, through one boundary class, and
treats everything between as opaque. It does **not** mean the mission is a pure function of
its inputs. It is not:

.. code-block:: python

   # openconcept/mission/phases.py -- five identical call sites
   self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"])

:class:`openconcept.mission.FullMissionWithReserve` takes an ``aircraft_model`` **class** as
an option and instantiates it inside every phase. OpenConcept therefore calls back into cdadt
twelve times per mission -- once per phase -- and the aircraft model it constructs is the
thing that turns flight conditions into forces.

So the boundary is a **contract**, not a wall. OpenConcept supplies flight conditions and
consumes forces and mass; cdadt supplies a group that converts one into the other.

.. figure:: /_static/blackbox.svg
   :align: center
   :alt: cdadt supplies a model class; OpenConcept instantiates it in every phase.

   The boundary is bidirectional: cdadt hands OpenConcept a class, and OpenConcept
   constructs it once per phase.

The contract
============

**What the aircraft model must produce**, in every phase, as promoted outputs:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Variable
     - Units
     - Meaning
   * - ``thrust``
     - N
     - Total thrust from all propulsors, along the flight path
   * - ``drag``
     - N
     - Total drag in the airplane axis, from all sources
   * - ``weight``
     - kg
     - Instantaneous aircraft **mass**. OpenConcept calls it ``weight``; the units are mass.

**What OpenConcept supplies**, and may be consumed, depends on the phase class. Flight
conditions from ``ComputeAtmosphericProperties`` (``fltcond|h``, ``T``, ``p``, ``rho``,
``a``, ``q``, ``M``, ``Utrue``, ``Ueas``) and the control inputs (``fltcond|CL``,
``throttle``, ``propulsor_active``) are present everywhere. Ground roll adds ``braking``.
The rotation phase, which uses Raymer's circular-arc transition rather than an integrated
ODE, has no ``fltcond|groundspeed`` or flight-path-angle vectors to offer. The engine-out
climb-angle check is a single condition, so it has no ``duration``.

The authoritative list is :func:`cdadt.mission.contract.phase_supplied_inputs`, and it is
verified against a really-built mission by ``cdadt/tests/mission/test_contract.py``.

Why the contract is declared as data
====================================

Because getting a name wrong does not raise.

OpenConcept adds the aircraft model with ``promotes_inputs=["*"], promotes_outputs=["*"]``.
A promoted input that matches nothing is legal in OpenMDAO: it simply holds its declared
default. So a provider that consumes ``fltcond|singamma`` during the rotation phase --
where it does not exist -- gets zero, and the mission converges. The result is wrong and
nothing reports it.

Two mechanisms address that:

1. Every name lives once, in :mod:`cdadt.mission.contract`, and the contract test iterates
   the **full** declared set against a real built mission. Over-declaring is the dangerous
   direction, so the test checks that every variable declared as supplied actually exists in
   that phase, and that its units are dimensionally identical to the declaration.
2. :class:`~cdadt.mission.aircraft_model.AircraftModelFactory` validates the discipline set
   against the contract **for every phase** at construction. A provider consuming a variable
   unavailable in some phase is rejected there, naming the phase.

The boundary is enforced, not just intended
===========================================

``cdadt/tests/mission/test_boundary.py`` scans cdadt's own source with an AST walk and fails
if any module other than :mod:`cdadt.mission.blackbox` imports ``openconcept.mission``. It
also asserts that the black box really does import it, so deleting the import cannot make the
check pass trivially, and that ``cdadt/core``, ``cdadt/certification``, ``cdadt/optimization``
and ``cdadt/io`` do not import OpenConcept at all.

Separately, ``cdadt/tests/test_openconcept_integrity.py`` fails if the OpenConcept working
tree has uncommitted modifications, or if any cdadt module assigns onto an OpenConcept symbol
(monkey-patching).

Injecting disciplines without global state
==========================================

OpenConcept constructs the aircraft model with a fixed signature and no argument through
which configured disciplines could be passed. Two workarounds are available and both are
rejected:

* A module-level registry that the class reads during ``setup()`` is global mutable state.
  Every sizing iteration and every optimizer function evaluation builds aircraft models, so
  results would depend on the order they were built.
* ``functools.partial`` works -- OpenConcept only calls the option -- but it is not a class,
  which makes the model tree harder to inspect.

:class:`~cdadt.mission.aircraft_model.AircraftModelFactory` instead builds a **new class per
discipline set**, carrying the disciplines as a class attribute:

.. code-block:: python

   factory = AircraftModelFactory([geometry, aerodynamics, propulsion, weights])
   model_class = factory.build()      # a genuine om.Group subclass, unique to this factory
   mission = FullMissionWithReserve(num_nodes=11, aircraft_model=model_class)

Two factories produce two distinct classes sharing nothing, and the base
:class:`~cdadt.mission.aircraft_model.CdadtAircraftModel` is never mutated.

Convergence is part of the interface
====================================

OpenConcept's mission is a coupled implicit system. Phase durations are solved by
``BalanceComp`` against altitude and range targets; throttle is solved for zero horizontal
acceleration at every node; V1 is solved so the continue and abort distances match. A Newton
solver started directly at the design mission will not generally converge.

:class:`~cdadt.mission.blackbox.MissionProfile` therefore carries an explicit
:class:`~cdadt.mission.blackbox.ContinuationStep` schedule -- progressively harder missions
run in order, each starting from the converged solution of the last:

.. code-block:: python

   MissionProfile(
       schedules={...},
       parameters={...},
       continuation=[
           ContinuationStep("short range at low altitude",
                            overrides={"mission_range": (500.0, "NM"),
                                       "cruise|h0": (5000.0, "ft")}),
       ],
   )

OpenConcept's own ``B738_sizing.py`` does this at lines 472-490, as a sequence of ``set_val``
calls in a run script. Making it a declared object means the schedule can be inspected,
tested, and reported, rather than being an undocumented incantation that a caller has to
remember.

Reading results
===============

Every result is declared once in :data:`cdadt.mission.contract.MISSION_OUTPUTS` with its
path and units, and :meth:`~cdadt.mission.blackbox.MissionBlackBox.read_outputs` returns all
of them. Reading the full set rather than the few a caller happens to want is what makes the
run report complete by construction.

Both balanced-field distances are declared, not just the one that becomes the §25.113
constraint. OpenConcept solves V1 so that they are equal, and reading both is how that solve
is verified after a run instead of assumed.

Total mission fuel is read from the **loiter** phase, because loiter is the last phase fuel
accumulates through. Reading it from an earlier phase would understate the fuel that closes
the ``MTOW = OEW + payload + fuel`` sizing loop -- and would still converge.

Replacing the black box
=======================

Swapping the mission analysis for something else means rewriting two modules,
:mod:`cdadt.mission.contract` and :mod:`cdadt.mission.blackbox`, and nothing else. The
boundary test is what keeps that true.
