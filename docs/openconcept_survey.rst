.. _openconcept_survey:

**********************************
Survey of the OpenConcept library
**********************************

cdadt depends on OpenConcept and treats it as read-only. This page records what that library
actually contains, and for every capability, whether cdadt uses it and why.

The purpose is falsifiability. "We used OpenConcept's mission analysis" is not a statement
anyone can check. A table saying *these* capabilities were adopted, *these* were examined and
rejected for *this* reason, and *these* do not apply to a tube-and-wing jet transport, is.
It also makes the boundary of the current work explicit: everything marked *available* is a
capability the framework could reach without new physics.

Surveyed at OpenConcept 1.2.6.

Mission analysis
================

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Capability
     - Status
     - Notes
   * - ``FullMissionWithReserve``
     - **Used**
     - The whole point. Balanced-field takeoff, climb/cruise/descent, Part 25 reserves,
       loiter. Wrapped by :class:`~cdadt.mission.blackbox.MissionBlackBox`.
   * - ``FullMissionAnalysis``
     - Available
     - Same without reserves. A design sized without reserves is not a Part 25 design, so
       cdadt uses the reserve variant.
   * - ``MissionWithReserve``
     - Available
     - Reserves but no balanced-field takeoff. Drops §25.113, which is a requirement cdadt
       exists to enforce.
   * - ``BasicMission``
     - Available
     - Climb/cruise/descent only. Useful for a cheaper analysis mode; not currently exposed.
   * - ``PhaseGroup`` / ``TrajectoryGroup``
     - **Used**
     - Indirectly but critically: ``PhaseGroup`` finds ``Integrator`` instances inside the
       aircraft model and links their states across phases. That is the mechanism by which
       fuel burned in climb carries into cruise.
   * - ``ClimbAnglePhase``
     - **Used**
     - Supplies the §25.121(b) engine-out climb condition at V₂.

Aerodynamics
============

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Capability
     - Status
     - Notes
   * - ``ParasiteDragCoefficient_JetTransport``
     - **Used**
     - Component drag buildup, clean and takeoff configuration.
   * - ``PolarDrag``
     - **Used**
     - Induced drag from the parabolic polar.
   * - ``CleanCLmax`` / ``FlapCLmax``
     - **Used**
     - Maximum lift, clean and flapped. The same correlation serves takeoff and landing, so
       the two cannot disagree about the same wing.
   * - ``StallSpeed``
     - **Used**
     - Landing-configuration stall speed at maximum landing weight.
   * - ``openaerostruct.AerostructDragPolar``
     - **Adopted**
     - Coupled aerostructural analysis. Outputs ``drag``, ``failure`` (KS-aggregated stress
       constraint) and ``ac|weights|W_wing``. Basis of the structures discipline; see below.
   * - ``openaerostruct.VLMDragPolar``
     - Available
     - Aerodynamics-only VLM. Superseded here by the aerostructural version, which gives the
       structural outputs as well.
   * - ``openaerostruct.CLmaxCriticalSectionVLM``
     - Available
     - Physics-based CLmax from the critical section, rather than the flap correlation. A
       clear upgrade path for the landing and takeoff speeds.
   * - ``ParasiteDragCoefficient_BWB``
     - Not applicable
     - Blended wing body. cdadt models a tube-and-wing transport.

Weights
=======

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Capability
     - Status
     - Notes
   * - ``JetTransportEmptyWeight``
     - **Used**
     - Component empty-weight buildup.
   * - ``weights_turboprop`` / ``weights_twin_hybrid``
     - Not applicable
     - Wrong aircraft class.
   * - ``weights_BWB``
     - Not applicable
     - Blended wing body.

Geometry, stability, atmospherics
=================================

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Capability
     - Status
     - Notes
   * - ``WingMACTrapezoidal``, ``CylinderSurfaceArea``
     - **Used**
     - Derived geometry.
   * - ``HStab`` / ``VStabVolumeCoefficientSizing``
     - **Used**
     - Empennage sizing by tail volume coefficient.
   * - ``WingSweepFromSections`` and the other section-based planform tools
     - Available
     - Multi-section planforms. cdadt currently models a trapezoidal wing.
   * - ``ComputeAtmosphericProperties``
     - **Used**
     - Indirectly: the mission phases build it and supply ``fltcond|*`` to the aircraft
       model.
   * - ``openaerostruct.mesh_gen``, ``wave_drag``
     - Available
     - Consumed internally by the aerostructural group.

Propulsion
==========

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Capability
     - Status
     - Notes
   * - ``RubberizedTurbofan`` (CFM56, N+3)
     - **Used**
     - Scalable engine deck, surrogate of pyCycle data.
   * - ``CFM56`` / ``N3`` / ``N3Hybrid`` decks
     - Available
     - Consumed by ``RubberizedTurbofan``; usable directly for a fixed-size engine.
   * - ``SimpleTurboshaft``, ``SimplePropeller``, ``SimpleMotor``, ``SimpleGenerator``,
       ``PowerSplit``
     - Not applicable
     - Turboprop and electric components.
   * - ``propulsion.systems`` (turboprop, series hybrid, all-electric)
     - Not applicable
     - Prebuilt systems for other configurations.

Not applicable to this aircraft
===============================

Examined and set aside. Each would matter for a different configuration, and the provider
abstraction is what makes them reachable without disturbing what exists.

``thermal`` (11 modules)
    Heat exchangers, ducts, pumps, chillers, heat pipes, battery and motor cooling. A
    conventional turbofan transport has no thermal management system to size.

``energy_storage``
    Batteries and liquid-hydrogen tanks with boil-off and structural models. Relevant to a
    hydrogen or electrified derivative, not a kerosene 737.

``costs.TurbopropOperatingCost``
    Operating cost, but commuter/turboprop-class correlations. Applying them to a jet
    transport would be extrapolation outside the fit.

``utilities.visualization``
    ``plot_trajectory``, ``plot_trajectory_grid``, ``plot_OAS_mesh``,
    ``plot_OAS_force_contours``. Not physics. cdadt has no plotting layer yet; these are
    what it should build on when it does, particularly the OAS mesh and force-contour plots
    now that the structures discipline exists.

Examples read
=============

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Example
     - What it contributed
   * - ``B738_sizing.py``
     - **The template.** The only example doing true sizing: empirical weight and drag
       buildups, rubberized engine, MTOW closure under a top-level Newton solver, and the
       staged continuation schedule needed to converge it. cdadt's architecture follows it.
       Also the source of the published golden values cdadt validates against.
   * - ``B738_aerostructural.py``
     - **Basis of the structures discipline.** Spar and skin thickness and twist as design
       variables, and a ``2_5g_KS_failure <= 0`` constraint — a structural failure
       constraint at the §25.337 limit maneuvering load factor.
   * - ``B738_VLM_drag.py``
     - Aerostructural wing optimization with mission analysis (Adler & Martins, *J.
       Aircraft*, 2022). Confirms the intended coupling between wing design and mission.
   * - ``B738.py``
     - The non-sizing B738: fixed weights rather than a buildup. Superseded by
       ``B738_sizing.py`` for cdadt's purposes.
   * - ``minimal.py``, ``minimal_integrator.py``
     - The documented minimum an aircraft model must supply, and how ``Integrator`` drives
       cross-phase fuel accumulation. Confirmed cdadt's reading of the interface contract.
   * - ``TBM850.py``, ``KingAirC90GT.py``, ``Caravan.py``
     - Turboprop models. Same aircraft-model interface, different physics — useful evidence
       that the contract in :mod:`cdadt.mission.contract` is configuration-independent.
   * - ``HybridTwin*.py``, ``ElectricSinglewithThermal.py``,
       ``N3_HybridSingleAisle_Refrig.py``
     - Hybrid-electric and thermal integration. Not applicable here, but they demonstrate
       the mission black box is agnostic to the propulsion system behind it — which is the
       property cdadt's provider abstraction depends on.

Consequences for cdadt
======================

Two things this survey changed.

**A structures discipline was missing and is now warranted.** ``AerostructDragPolar`` gives a
structurally sized wing, a KS-aggregated failure constraint, and a computed wing weight —
replacing the empirical wing-weight correlation with something a load case actually drives.
That also supplies a certification requirement cdadt previously had no way to express:
14 CFR 25.337's positive limit maneuvering load factor, enforced under §25.305.

**Aerostructural coupling is not separable.** ``AerostructDragPolar`` is one analysis
producing ``drag``, ``failure`` and ``ac|weights|W_wing`` together. Modeling aerodynamics and
structures as two disciplines that each build their own copy would run the coupled solve
twice and let the two answers drift apart. They must share one provider instance. That is a
statement about the physics, not a convenience: in an aerostructural wing, the loads depend
on the deflected shape and the deflected shape depends on the loads.
