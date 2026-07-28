.. _tutorials:

*********
Tutorials
*********

.. contents::
   :local:
   :depth: 1

Size an aircraft
================

Sizing solves ``MTOW = OEW + payload + fuel``, where the fuel comes back from the mission
that the takeoff weight determines.

.. code-block:: python

   from cdadt.aircraft import jet_transport_disciplines
   from cdadt.core import AircraftConfiguration
   from cdadt.mission import SizingLoop

   config = AircraftConfiguration.from_yaml("configs/b738.yml")
   loop = SizingLoop(
       jet_transport_disciplines(config, engine_deck="CFM56"),
       config,
       num_nodes=11,
   )

   problem = loop.build()
   loop.converge(problem)

   results = loop.results(problem)
   print(f"MTOW  {results['MTOW']:10.1f} kg")
   print(f"OEW   {results['OEW']:10.1f} kg")
   print(f"fuel  {results['total_fuel']:10.1f} kg")

:meth:`~cdadt.mission.sizing.SizingLoop.converge` seeds the starting state, runs the
continuation schedule from the configuration, and then runs the design mission. If the
Newton solver fails, it **raises** rather than returning: a diverged mission still produces
numbers — a negative field length, a range that misses the one requested — and they are
indistinguishable from results.

Optimize against a certification basis
======================================

.. code-block:: python

   from cdadt.certification import (
       ApproachSpeedLimit, BalancedFieldLength, CertificationBasis,
       EngineOutClimbGradient, LandingFieldLengthLimit, ThrottleMargin,
   )
   from cdadt.optimization import DesignProblem, IpoptDriver

   certification = CertificationBasis([
       BalancedFieldLength(config),
       EngineOutClimbGradient(config),
       LandingFieldLengthLimit(config),
       ApproachSpeedLimit(config),
       *[ThrottleMargin(config, phase=p) for p in ("climb", "cruise", "descent")],
   ])

   design_problem = DesignProblem(
       disciplines=jet_transport_disciplines(config, engine_deck="CFM56"),
       config=config,
       certification=certification,
       driver=IpoptDriver(config),
       objective="total_fuel",
   )

   problem = design_problem.build()
   result = design_problem.run(problem)
   print(result.report())

   if not result.optimal:
       raise SystemExit(f"not an optimum: {result.outcome.status}")

That last check is not boilerplate. ``run_driver`` reports that it returned, not that it
converged; see :ref:`optimization`.

Add a discipline provider
=========================

A provider says *how* a discipline computes what it declares. To swap the empirical drag
buildup for something else, write a provider and put it in the discipline:

.. code-block:: python

   from cdadt.core.provider import Provider
   from cdadt.core.variables import Variable, VariableSet

   class MyDragProvider(Provider):
       @property
       def name(self):
           return "my_drag"

       @property
       def reference(self):
           # Appears in the run report next to every result this influenced.
           return "Smith & Jones 2019, Eq. 14, corrected per the 2021 erratum."

       def provides(self):
           return VariableSet([Variable("drag", "N", vectorized=True)])

       def requires(self):
           return VariableSet([
               Variable("fltcond|q", "N/m**2", vectorized=True),
               Variable("fltcond|CL", None, vectorized=True),
               Variable("ac|geom|wing|S_ref", "m**2"),
           ])

       def validate_configuration(self):
           # Fails at construction, listing everything absent at once.
           self.config.require_all(["ac|aero|my_method|coefficient"])

       def build(self, group, num_nodes, flight_phase):
           ...

   Aerodynamics(MyDragProvider(config), config)

Three things are worth getting right, because each has a silent failure mode:

**Declare everything you consume, including inputs with component-level defaults.** This is
not bookkeeping. While building cdadt's own drag provider, ``ac|aero|polar|e`` was left out
of ``requires()``; the model then computed induced drag from ``PolarDrag``'s own default
instead of the configured 0.801, and every drag number was low by about 7% with nothing
failing.

**Declare units honestly.** They are what
:func:`~cdadt.core.discipline.resolve_input_units` uses to reconcile components that
disagree, which the wrapped OpenConcept components legitimately do.

**Branch on** ``flight_phase`` **if the configuration changes.** The four takeoff phases run
with flaps deployed; running clean drag through them understates both the field length and
the §25.121 climb gradient, and the mission converges anyway.

Add a certification requirement
===============================

.. code-block:: python

   from cdadt.certification.basis import Requirement, Sense

   class MyRequirement(Requirement):
       LIMIT = "certification|my_rule|limit"

       @property
       def name(self):
           return "far25_xxx_my_rule"

       @property
       def regulation(self):
           return "14 CFR 25.xxx"

       @property
       def title(self):
           return "What the rule requires"

       @property
       def sense(self):
           return Sense.UPPER      # or LOWER

       @property
       def units(self):
           return "ft"

       def validate_configuration(self):
           self.config.require_all([self.LIMIT])

       def limit(self):
           return self.config.scalar(self.LIMIT, units=self.units)

       @property
       def limit_source(self):
           return self.config.source(self.LIMIT)   # cited in the traceability matrix

       def constrained_path(self, blackbox):
           return blackbox.path("some_declared_output")

Ask the black box for the path; never hardcode an OpenConcept one. If the quantity is not
already declared, add it to :mod:`cdadt.mission.contract` — that is what keeps the contract
tests able to verify it.

If the requirement needs physics the mission does not model, put that physics in a
**provider** and have the requirement read it, as the landing requirements do. A requirement
that also computes its own value entangles the regulation with the method, and the boundary
test enforces the separation.

Then write the test that matters: perturb the limit past the value the design achieves, and
assert the requirement flips to not-met. A requirement that cannot be made to bind is not
connected to anything, and it will report a margin regardless.

Change the mission or the aircraft
==================================

Both live in the configuration, not in code. To fly a different mission, edit the
``mission:`` section — range, altitudes, reserves, the per-phase speed schedules, and the
continuation schedule that gets the solver there. To design a different aircraft, edit
``ac:``.

If the mission stops converging after a change, the continuation schedule is usually the
place to look: add an intermediate step that reaches the new condition gradually. That is
what the existing two steps do for the design range and the steep initial descent.
