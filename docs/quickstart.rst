Quickstart
==========

Size an aircraft
----------------

.. code-block:: bash

   cdadt size cases/b738.yaml

Converges the 737-800 case against its 2800 nmi design mission with a Part 25 reserve diversion
and loiter, and prints every response of every discipline. Roughly five seconds at 21 nodes per
phase.

.. code-block:: text

   Black box : openconcept.examples.B738_sizing:B738SizingMissionAnalysis
   Grid      : 21 nodes per phase

   weights
   -------
     MTOW                               78345.6435  kg
     OEW                                41748.3258  kg
     MLW                                62676.5148  kg
     ...

   performance
   -----------
     block_fuel                         15977.0628  kg
     total_fuel                         18597.3177  kg
     takeoff_field_length                5247.7948  ft
     abort_distance                      5247.7948  ft
     V1                                   135.0566  kn
     V2                                   155.4331  kn
     engine_out_climb_gradient              0.0579  rad
     ...

``takeoff_field_length`` and ``abort_distance`` are equal because the black box solves the
decision speed V\ :sub:`1` to make them so. A run where they differ has not converged.

Optimize against a certification basis
--------------------------------------

.. code-block:: bash

   cdadt optimize cases/b738_optimization.yaml

Minimizes fuel with reserves over the wing planform and the engine rating, subject to the
balanced field length, the engine-out second-segment climb gradient and the throttle limits of
the engine deck. Prints the design variables, a baseline-to-optimum comparison of every result,
and the traceability matrix:

.. code-block:: text

   regulation               requirement                                          value       limit      margin  units  status
   14 CFR 25.113            Balanced field length within the runway available  6263.71     8000.00     1736.29  ft     MET
   14 CFR 25.121(b)(1)(i)   OEI second-segment climb gradient                     0.0554      0.0240      0.0314 rad   MET
   design                   Throttle within the engine deck's range in climb      1.0000      1.0000     -0.0000 -     ACTIVE
   design                   Throttle within the engine deck's range in cruise     0.8287      1.0000      0.1713 -     MET

   4 of 4 requirements met, 1 active, 0 violated.

Read the interface
------------------

.. code-block:: bash

   cdadt inspect cases/b738.yaml --what inputs --filter "ac|geom|wing"

Prints what the case's black box accepts and what it publishes, without running anything. This
is the interface reference, generated rather than transcribed. See :doc:`interface`.

From Python
-----------

The command line is a thin wrapper. The API underneath is three lines:

.. code-block:: python

   from cdadt import Config, SizingAnalysis

   analysis = SizingAnalysis(Config.from_yaml("cases/b738.yaml"))
   results = analysis.run()
   print(results["MTOW"], results["total_fuel"])

and for an optimization:

.. code-block:: python

   from cdadt import Config, Optimizer, SizingAnalysis

   optimizer = Optimizer(SizingAnalysis(Config.from_yaml("cases/b738_optimization.yaml")))
   outcome = optimizer.run()
   print(optimizer.report(outcome))

Where to go next
----------------

- :doc:`configuration` -- every key of the case file.
- :doc:`architecture` -- the classes, and why the boundary is drawn where it is.
- :doc:`tutorials` -- change the aircraft, add a requirement, add a discipline.
- :doc:`validation` -- what has been established, and what has not.
