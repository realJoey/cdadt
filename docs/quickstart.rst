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

Size it on your own aerodynamics
---------------------------------

The drag can be cdadt's rather than OpenConcept's, computed from a vortex lattice built on the wing
the case file describes. Everything else -- the balanced field, the reserves, the engine deck, the
weight closure -- stays OpenConcept's.

.. code-block:: bash

   cdadt size cases/b738_avl.yaml     # openavl          (needs: pip install -e ".[avl]")
   cdadt size cases/b738_oas.yaml     # OpenAeroStruct   (needs: pip install -e ".[transonic]")

Six cases ship, in three sets of two -- an analysis and an optimization each for the aircraft
configuration alone, and for the same configuration with each lattice supplying the aerodynamic
loads:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Aerodynamics
     - Sizing
     - Optimization
   * - OpenConcept's own
     - ``b738.yaml``
     - ``b738_optimization.yaml``
   * - openavl vortex lattice
     - ``b738_avl.yaml``
     - ``b738_avl_optimization.yaml``
   * - OpenAeroStruct vortex lattice
     - ``b738_oas.yaml``
     - ``b738_oas_optimization.yaml``

What changes, and what does not:

.. code-block:: text

                          reference     openavl    OpenAeroStruct
   MTOW (kg)               78,345.6    76,827.2         76,554.7
   Fuel with reserves      18,597.3    17,402.7         17,188.7
   Balanced field (ft)      5,247.8     4,986.0          4,945.3
   wing_span (m)          not published   34.3143          34.3143

Two things to know before reading that table. The lattices report a span efficiency near 0.99
against the 0.801 the case file assumes, which is most of the difference -- but they also carry
transonic drag rise, which the reference has no way to model, and that pushes the other way. The
figures net the two. :doc:`aerodynamics` separates them.

And ``wing_span`` appears only for the lattice cases, because it is an *optional* response:
OpenConcept's own group never computes a span. A run that cannot report something says so under
"Not published by this black box" rather than omitting it silently.

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

Every run leaves the files a review asks for
---------------------------------------------

Both commands above already did this -- there is no flag. Each invocation writes its own
directory under ``run_outputs/``, named for the case and the moment it ran::

   run_outputs/b738_20260728_201512_out/
     report.txt      results.json    n2.html
     mission.pdf     trajectory.pdf  takeoff.pdf
     .openmdao_out   reports/

``mission.pdf`` reproduces ``B738_sizing.py``'s own figure and ``trajectory.pdf`` reproduces
``B738.py``'s; ``takeoff.pdf`` draws the balanced field, which neither example plots. An
optimization additionally leaves ``IPOPT.out``, the optimizer's own log. See :doc:`artifacts`.

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
