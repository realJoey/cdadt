The OpenConcept survey
======================

Choosing which OpenConcept analysis to drive is the single most consequential decision in this
repository: it fixes what physics is available, what can be constrained, and what the results
mean. It was therefore made by reading the library rather than by picking the obvious example.

This page records that survey -- roughly 30,000 lines across 100 source files -- and the
conclusion it reached, which is stronger than "this example looks suitable".

The conclusion
--------------

``openconcept.examples.B738_sizing:B738SizingMissionAnalysis`` is the **only** analysis in
OpenConcept that satisfies the requirement cdadt exists to meet. It is not the most convenient
choice among several; it is the only one.

Three properties are needed, and exactly one example has all three:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Example
     - Mission profile
     - Closes a weight loop
     - Scalable engine
   * - **B738_sizing.py**
     - **FullMissionWithReserve**
     - **yes**
     - **yes**
   * - B738.py
     - MissionWithReserve
     - no
     - no
   * - B738_VLM_drag.py
     - MissionWithReserve
     - no
     - no
   * - B738_aerostructural.py
     - BasicMission
     - no
     - no
   * - Caravan.py
     - FullMissionAnalysis
     - no
     - no
   * - KingAirC90GT.py
     - FullMissionAnalysis
     - no
     - no
   * - TBM850.py
     - FullMissionAnalysis
     - no
     - no
   * - HybridTwin.py, HybridTwin_thermal.py
     - FullMissionAnalysis
     - no
     - no
   * - HybridTwin_active_thermal.py
     - BasicMission
     - no
     - no
   * - ElectricSinglewithThermal.py
     - FullMissionAnalysis
     - no
     - no
   * - N3_HybridSingleAisle_Refrig.py
     - BasicMission
     - no
     - no
   * - minimal.py, minimal_integrator.py
     - BasicMission
     - no
     - no

Established by inspection of every file in ``openconcept/examples/``: grepping each for its
mission profile, for an empty-weight buildup group, and for ``RubberizedTurbofan``. Only
``B738_sizing.py`` matches on all three, and it is the only file in the library that imports
``RubberizedTurbofan`` at all.

Why each property is required
------------------------------

**The mission profile must include both balanced-field takeoff and Part 25 reserves.** The four
profiles in ``openconcept/mission/profiles.py`` are one class with two switches:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Profile
     - Takeoff (BFL)
     - Reserves
     - Suitable
   * - ``BasicMission``
     - no
     - no
     - no
   * - ``FullMissionAnalysis``
     - yes
     - no
     - no
   * - ``MissionWithReserve``
     - no
     - yes
     - no
   * - ``FullMissionWithReserve``
     - **yes**
     - **yes**
     - **yes**

Only the last has both, and the requirement was "takeoff balanced field length and mission plus
reserves". The other three are the same class constructed with ``include_takeoff`` or
``include_reserve`` switched off.

**It must close a weight loop.** A fixed-weight analysis answers "how much fuel does *this*
aeroplane burn". A sizing analysis answers "how big must the aeroplane be", by driving
:math:`\mathrm{MTOW} = \mathrm{OEW}(\mathrm{MTOW}) + W_\mathrm{payload} + W_\mathrm{fuel}(\mathrm{MTOW})`
to consistency. Only ``B738_sizing.py`` does this; every other example takes weights as data.
Without it, changing the wing would change the drag but not the structure, and the optimization
this repository performs would be meaningless.

**The engine must be scalable.** ``RubberizedTurbofan`` scales thrust and fuel flow from a fixed
CFM56 deck by a rated-thrust input, which is what makes engine size a design variable. The other
examples use fixed decks, surrogate maps for one specific engine, or propeller/electric systems.

What is inside the chosen box
-----------------------------

Following the imports of ``B738_sizing.py`` down, the modules that actually execute:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Module
     - What it contributes
   * - ``mission/profiles.py``
     - ``FullMissionWithReserve``: the twelve phases and their linking
   * - ``mission/phases.py``
     - Ground roll, rotation, climb-angle and steady-flight phases; the implicit V\ :sub:`1` solve
   * - ``mission/mission_groups.py``
     - Phase, integrator and trajectory group machinery
   * - ``aerodynamics/drag_jet_transport.py``
     - Component-by-component parasite drag buildup
   * - ``aerodynamics/aerodynamics.py``
     - ``PolarDrag``: the parabolic drag polar
   * - ``aerodynamics/CL_max_estimation.py``
     - Clean and flapped maximum lift coefficients
   * - ``propulsion/rubberized_turbofan.py``, ``propulsion/cfm56.py``
     - The scaled CFM56 thrust and fuel-flow deck
   * - ``weights/weights_jet_transport.py``
     - Structure, gear, nacelle and equipment weight correlations
   * - ``stability/tail_volume_coefficient_sizing.py``
     - Horizontal and vertical tail area from tail volume coefficient
   * - ``geometry/wing_planform.py``, ``geometry/wetted_area.py``
     - Mean aerodynamic chord, wetted areas
   * - ``atmospherics/``
     - Standard atmosphere, true airspeed, dynamic pressure, Mach
   * - ``utilities/math/integrals.py``
     - Simpson's rule integration of fuel burn -- the reason ``num_nodes`` must be odd

What is in OpenConcept and **not** used
----------------------------------------

Worth recording, because it bounds what cdadt could ever be asked to do without changing the box:

- ``aerodynamics/openaerostruct/`` (3,400 lines) -- VLM and aerostructural drag polars. Higher
  fidelity, and a different box. Not loaded.
- ``thermal/`` (5,100 lines) -- ducts, heat exchangers, heat pipes, chillers, pumps. For
  electrified propulsion thermal management.
- ``energy_storage/`` -- batteries and liquid-hydrogen tanks.
- ``propulsion/systems/`` -- series hybrid, all-electric and turboprop propulsion systems;
  ``N3.py``, ``motor.py``, ``generator.py``, ``propeller.py``, ``turboshaft.py``.
- ``weights/weights_BWB.py``, ``weights_turboprop.py``, ``weights_twin_hybrid.py`` -- other
  aircraft classes.
- ``aerodynamics/drag_BWB.py`` -- blended wing body drag.
- ``costs/costs_commuter.py`` -- a commuter-aircraft cost model.
- ``utilities/visualization.py`` -- OpenConcept's own plotting helpers.

None of it is imported by cdadt, and a contract test proves that by checking
:data:`sys.modules` after a box is built. Any of it could become reachable by naming a different
analysis in ``black_box.model`` -- which is configuration, not code. See :doc:`blackbox`.

How the survey was performed
-----------------------------

Reproducible, and worth repeating whenever OpenConcept is updated:

.. code-block:: bash

   # every source file and its size
   find openconcept -name '*.py' -not -path '*/tests/*' | sort | xargs wc -l

   # which mission profile each example uses
   grep -oE "FullMissionWithReserve|FullMissionAnalysis|MissionWithReserve|BasicMission" \
       openconcept/examples/*.py | sort -u

   # which examples close a weight loop
   grep -l "EmptyWeight" openconcept/examples/*.py

   # which examples can scale an engine
   grep -l "RubberizedTurbofan" openconcept/examples/*.py

   # what a built box actually loads
   cdadt inspect cases/b738.yaml

The last command is the authoritative one: it reads the interface off the live model rather than
off this page. See :doc:`interface`.
