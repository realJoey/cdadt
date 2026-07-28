Tutorials
=========

Four things you are likely to want to do.

1. Size an aircraft
-------------------

.. code-block:: python

   from cdadt import AircraftDefinition, MissionProfile, SizingAnalysis

   analysis = SizingAnalysis(
       aircraft=AircraftDefinition.from_yaml("cases/b738_aircraft.yaml"),
       profile=MissionProfile.from_yaml("cases/b738_mission.yaml"),
       num_nodes=21,
   )
   results = analysis.run(verbose=True)
   print(analysis.report())

   print(results["MTOW"])                   # 78345.64 kg
   print(results["total_fuel"])             # 18597.32 kg, including reserves
   print(results["takeoff_field_length"])   # 5247.79 ft

Or from the command line:

.. code-block:: bash

   python examples/size_b738.py
   python examples/size_b738.py --num-nodes 11 --quiet

``num_nodes`` must be odd -- fuel burn is integrated with Simpson's rule, and an even count is
rejected rather than silently mis-integrated.

2. Change the aircraft
----------------------

Edit ``cases/b738_aircraft.yaml``, or set values in code:

.. code-block:: python

   aircraft = AircraftDefinition.from_yaml("cases/b738_aircraft.yaml")
   aircraft.set("ac|geom|wing|AR", 11.0)
   aircraft.set("ac|propulsion|engine|rating", 24000.0)   # in its stored units, lbf

:meth:`~cdadt.aircraft.AircraftDefinition.set` refuses names that are not already defined, so
a typo cannot quietly create a variable nothing reads. Note also that quantities the model
*computes* -- tail areas, MAC, OEW, MTOW -- must not appear in the definition; see
:doc:`architecture`.

3. Optimize against a certification basis
-----------------------------------------

.. code-block:: python

   from cdadt import (
       BalancedFieldLength, CertificationBasis, DesignOptimizer, DesignVariable,
       EngineOutClimbGradient, Part25PhaseModel, SizingAnalysis, ThrottleMargin,
   )

   analysis = SizingAnalysis(aircraft, profile, phase_model=Part25PhaseModel, num_nodes=11)

   optimizer = DesignOptimizer(
       analysis=analysis,
       design_variables=[
           DesignVariable("ac|geom|wing|S_ref", lower=90.0, upper=180.0, units="m**2"),
           DesignVariable("ac|geom|wing|AR", lower=7.0, upper=13.0),
           DesignVariable("ac|propulsion|engine|rating", lower=18000.0, upper=34000.0, units="lbf"),
       ],
       objective="total_fuel",
       certification=CertificationBasis([
           BalancedFieldLength(limit=8000.0, source="8000 ft dry runway, sea level ISA"),
           EngineOutClimbGradient(limit=0.024, source="14 CFR 25.121(b)(1)(i), twin"),
           ThrottleMargin(phase="climb", limit=1.0, source="engine deck rated condition"),
       ]),
       optimizer="IPOPT",
   )
   optimizer.run()
   print(optimizer.report())

Or:

.. code-block:: bash

   python examples/optimize_b738.py
   python examples/optimize_b738.py --objective MTOW --optimizer SLSQP

4. Add a requirement
--------------------

If the mission already produces the quantity, a requirement is six properties:

.. code-block:: python

   from cdadt.certification import Requirement, Sense

   class CruiseMachMinimum(Requirement):
       """The aircraft must be able to hold its block speed."""

       @property
       def name(self): return "cruise_mach_minimum"

       @property
       def regulation(self): return "operator requirement"

       @property
       def title(self): return "Cruise Mach at or above block speed"

       @property
       def sense(self): return Sense.LOWER

       @property
       def units(self): return None

       @property
       def path(self): return "mission.cruise.fltcond|M"

   basis = CertificationBasis([CruiseMachMinimum(limit=0.78, source="Network block-time study")])

If the mission does **not** produce it, also override
:meth:`~cdadt.certification.Requirement.build` to add the analysis, as
:class:`~cdadt.certification.ApproachSpeed` does for the landing it has to fly itself.

Both the limit and its ``source`` are required arguments. That is deliberate; see
:doc:`certification`.

5. Swap a discipline
--------------------

Both models list their disciplines as a class attribute:

.. code-block:: python

   from cdadt import JetTransportPhaseModel
   from cdadt.disciplines import MassProperties, Propulsion

   class HydrogenPhaseModel(JetTransportPhaseModel):
       disciplines = (Aerodynamics, HydrogenPropulsion, MassProperties)

   analysis = SizingAnalysis(aircraft, profile, phase_model=HydrogenPhaseModel)

A new discipline subclasses :class:`~cdadt.disciplines.base.PhaseDiscipline` or
:class:`~cdadt.disciplines.base.AircraftDiscipline`, sets ``discipline_name``, and implements
``setup``. It must promote the names its neighbours promote -- that is the whole coupling
mechanism -- and it must not inherit from an OpenConcept class, which
:mod:`tests.test_openconcept_boundary` enforces.
