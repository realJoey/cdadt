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
balanced field length, the engine-out second-segment climb gradient and the throttle band of the
engine deck. Prints the design variables, a baseline-to-optimum comparison of every result, and
the traceability matrix:

.. code-block:: text

   regulation              constraint                                             value        bound      margin  units  status
   ---------------------------------------------------------------------------------------------------------------------------
   14 CFR 25.113           Balanced field length within the runway available  6587.1294      <= 8000  1412.8706   ft     MET
   14 CFR 25.121(b)(1)(i)  OEI second-segment climb gradient                     0.0501     >= 0.024     0.0261   rad    MET
   -                       Climb throttle within the engine deck                 1.0500 0.01 to 1.05     0.0000   -      ACTIVE
   -                       Cruise throttle within the engine deck                0.8691 0.01 to 1.05     0.1809   -      MET

   Where each limit came from
   --------------------------
     takeoff_field_length (14 CFR 25.113): Design field length, 8000 ft dry runway at sea level, ISA
     engine_out_climb_gradient (14 CFR 25.121(b)(1)(i)): Two-engine aeroplane, 2.4% second-segment minimum

   Design constraints with no stated regulation or source: climb_throttle, cruise_throttle.
   These bound the design; they are not certification evidence.

   4 of 4 constraints met, 1 active, 0 violated.

Leave the files a review asks for
---------------------------------

.. code-block:: bash

   cdadt size cases/b738.yaml --outputs b738_out

Writes ``n2.html`` (OpenMDAO's diagram of the model that was actually run), ``trajectory.pdf``
(what was flown, against range), ``report.txt`` and ``results.json`` into one directory. Works on
``optimize`` too. See :doc:`artifacts`.

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
- :doc:`artifacts` -- the files a run leaves behind.
- :doc:`architecture` -- the classes, and why the boundary is drawn where it is.
- :doc:`tutorials` -- change the aircraft, free a variable, add a constraint, add a discipline.
- :doc:`validation` -- what has been established, and what has not.
