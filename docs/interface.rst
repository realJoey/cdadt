The interface reference
=======================

The complete list of what the black box accepts and what it publishes is **generated**, not
transcribed:

.. code-block:: bash

   cdadt inspect cases/b738.yaml
   cdadt inspect cases/b738.yaml --what inputs
   cdadt inspect cases/b738.yaml --what outputs --filter fuel_burn

That is deliberate. A hand-written interface table in documentation is correct on the day it is
written and slowly stops being correct afterwards, and there is no way to tell by reading it.
``inspect`` builds the case's box on a three-node grid -- a fraction of a second, nothing is
converged -- and reads the interface off the live model.

What counts as an input
-----------------------

A variable is settable if it is an **independent variable** of the box: an output of one of its
``IndepVarComp``\ s, or an input no component drives. Those are exactly the variables a case
file may set and a driver may move as a design variable.

For the shipped case that is 241 variables, of which 34 are ``ac|`` design parameters and 8 are
mission-level parameters; the rest are per-phase settings, most importantly the
equivalent-airspeed and vertical-speed schedules and the ISA temperature increment of each
phase.

.. code-block:: text

   $ cdadt inspect cases/b738.yaml --what inputs --filter "ac|geom|wing"

   Inputs the box accepts (5 of 241 shown)
   ------------------------------------------------------------------------
     ac|geom|wing|AR                                      -          (1,)
     ac|geom|wing|S_ref                                   m**2       (1,)
     ac|geom|wing|c4sweep                                 deg        (1,)
     ac|geom|wing|taper                                   -          (1,)
     ac|geom|wing|toverc                                  -          (1,)

Note what is *not* in that list: ``ac|geom|wing|MAC``. The mean aerodynamic chord is computed by
the box from the planform, so it is an output. The same is true of ``ac|geom|hstab|S_ref``,
``ac|geom|vstab|S_ref``, ``ac|geom|fuselage|S_wet``, ``ac|aero|CLmax_cruise``,
``ac|aero|CLmax_TO``, ``ac|weights|OEW``, ``ac|weights|MLW`` and ``ac|weights|MTOW``. Attempting
to set one is an error naming it, with suggestions.

What counts as an output
------------------------

Everything the box produces -- 745 outputs for the shipped case at three nodes, since most of
them are per-phase and per-node. Promoted *inputs* are omitted from the listing: a promoted
input name can address several components at once and so is not unambiguously readable. Every
quantity a discipline reports is an output.

The named responses
-------------------

cdadt gives short names to the outputs its disciplines report, so a case file can write
``objective: {name: total_fuel}`` rather than
``mission.loiter.fuel_burn_integ.fuel_burn_final``. Those names are also what
:doc:`certification` requirements are evaluated on. They come from the discipline classes
themselves, so the list below is generated from the same source the code uses:

.. list-table::
   :header-rows: 1
   :widths: 18 26 34 12 10

   * - Discipline
     - Response
     - Path in the box
     - Units
     - Optional
   * - geometry
     - ``wing_MAC``
     - ``ac|geom|wing|MAC``
     - m
     -
   * - geometry
     - ``fuselage_wetted_area``
     - ``ac|geom|fuselage|S_wet``
     - m**2
     -
   * - geometry
     - ``nacelle_wetted_area``
     - ``ac|geom|nacelle|S_wet``
     - m**2
     -
   * - geometry
     - ``tail_lever_arm``
     - ``tail_lever_arm_estimate.c4_to_wing_c4``
     - m
     - yes
   * - aerodynamics
     - ``CLmax_cruise``
     - ``ac|aero|CLmax_cruise``
     - \-
     -
   * - aerodynamics
     - ``CLmax_takeoff``
     - ``ac|aero|CLmax_TO``
     - \-
     -
   * - propulsion
     - ``engine_weight``
     - ``empty_weight.single_engine.W_engine``
     - kg
     - yes
   * - propulsion
     - ``engines_weight``
     - ``empty_weight.W_engines``
     - kg
     - yes
   * - propulsion
     - ``thrust_reverser_weight``
     - ``empty_weight.W_thrust_rev``
     - kg
     - yes
   * - propulsion
     - ``fuel_system_weight``
     - ``empty_weight.W_fuelsystem``
     - kg
     - yes
   * - stability
     - ``hstab_area``
     - ``ac|geom|hstab|S_ref``
     - m**2
     -
   * - stability
     - ``vstab_area``
     - ``ac|geom|vstab|S_ref``
     - m**2
     -
   * - structures
     - ``structure_weight``
     - ``empty_weight.W_structure``
     - kg
     - yes
   * - structures
     - ``wing_weight``
     - ``empty_weight.W_wing``
     - kg
     - yes
   * - structures
     - ``fuselage_weight``
     - ``empty_weight.W_fuselage``
     - kg
     - yes
   * - weights
     - ``MTOW``
     - ``ac|weights|MTOW``
     - kg
     -
   * - weights
     - ``OEW``
     - ``ac|weights|OEW``
     - kg
     -
   * - weights
     - ``MLW``
     - ``ac|weights|MLW``
     - kg
     -
   * - performance
     - ``block_fuel``
     - ``mission.descent.fuel_burn_integ.fuel_burn_final``
     - kg
     -
   * - performance
     - ``total_fuel``
     - ``mission.loiter.fuel_burn_integ.fuel_burn_final``
     - kg
     -
   * - performance
     - ``takeoff_field_length``
     - ``mission.bfl.distance_continue``
     - ft
     -
   * - performance
     - ``abort_distance``
     - ``mission.bfl.distance_abort``
     - ft
     -
   * - performance
     - ``V1``
     - ``mission.takeoff|v1``
     - kn
     -
   * - performance
     - ``V2``
     - ``mission.engineoutclimb.takeoff|v2``
     - kn
     -
   * - performance
     - ``engine_out_climb_gradient``
     - ``mission.engineoutclimb.gamma``
     - rad
     -
   * - performance
     - ``mission_range_flown``
     - ``mission.descent.ode_integ_phase.range_final``
     - nmi
     -
   * - performance
     - ``climb_throttle``
     - ``mission.climb.throttle``
     - \-
     -

The table is abridged -- the equipment weight breakdown, the remaining throttle histories and
the phase durations are omitted for length. The authoritative list is:

.. code-block:: python

   from cdadt import ResponseCatalog

   catalog = ResponseCatalog()
   for name in sorted(catalog):
       response = catalog.response(name)
       print(f"{catalog.owner(name):14s} {name:26s} {response.path:52s} {response.units}")

Optional responses
------------------

A response marked optional exists only in a particular black-box model. The structural and
equipment weight breakdowns are optional because they exist as long as the box's weight model
publishes them -- which OpenConcept's jet-transport buildup does, and a different sizing model
need not.

A missing optional response is reported as unavailable in the run report and in the archived
JSON, rather than being dropped. A missing *required* response is an error: it means the box is
not the model the discipline was written against, and returning a partial result quietly would
let a report claim a quantity it never read.
