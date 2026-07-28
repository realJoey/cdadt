Architecture
============

cdadt is eight classes and a boundary. This page is the map.

.. code-block:: text

    AircraftDefinition          MissionProfile
    (ac| design parameters)     (range, altitudes, speed schedules,
             |                   continuation schedule)
             |                          |
             +-------------+------------+
                           |
                     SizingAnalysis  ......... owns the OpenMDAO Problem,
                           |                   the Newton solver, and the run
                           v
                       SizingModel
       +-------------------+-------------------+
       |                   |                   |
   design_parameters   aircraft            takeoff_weight     mission
   (IndepVarComp)      disciplines         (MTOW closure)     (OpenConcept,
                       Geometry                               a black box)
                       Stability                                  |
                       HighLift                        instantiates per phase
                       Weights                                    v
                                                        JetTransportPhaseModel
                                                            Aerodynamics
                                                            Propulsion
                                                            MassProperties

    DesignOptimizer ......... design variables + objective + CertificationBasis,
                              registered on the model before setup

Disciplines
-----------

Every discipline is a class -- specifically an :class:`openmdao.api.Group` subclass -- that
owns the analysis for one engineering domain and nothing else. It adds its own subsystems,
declares its own promotions, and never reaches into a sibling. Coupling happens because
disciplines promote the same variable names, which is how OpenMDAO connects them and how
OpenConcept's phases connect to them.

There is no global state anywhere in the package. Everything a discipline needs arrives as an
OpenMDAO option or a promoted input. That is not a style preference: the optimizer rebuilds
and re-evaluates this model dozens of times in one process, and a module-level cache would
make iteration *n* depend on iteration *n-1*.

Two scopes, and the difference is physical
------------------------------------------

:class:`~cdadt.disciplines.base.AircraftDiscipline`
    Evaluated once, above the mission. Wing MAC, tail areas, operating empty weight and
    maximum lift coefficients are properties of the design. They do not change between cruise
    and descent, and evaluating them inside each phase would allow them to.

:class:`~cdadt.disciplines.base.PhaseDiscipline`
    Evaluated at every analysis node of every phase. Drag, thrust and the fuel-burn
    bookkeeping that carries mass through the mission are functions of the flight condition.
    OpenConcept builds these itself, once per phase, sized to that phase.

======================================================== ========= ===========================
Discipline                                               Scope     Provides
======================================================== ========= ===========================
:class:`~cdadt.disciplines.geometry.Geometry`            Aircraft  MAC, tail lever arms, wetted areas
:class:`~cdadt.disciplines.stability.Stability`          Aircraft  Horizontal and vertical tail areas
:class:`~cdadt.disciplines.high_lift.HighLift`           Aircraft  ``CLmax_cruise``, ``CLmax_TO``
:class:`~cdadt.disciplines.weights.Weights`              Aircraft  OEW, MLW
:class:`~cdadt.disciplines.aerodynamics.Aerodynamics`    Phase     ``drag``
:class:`~cdadt.disciplines.propulsion.Propulsion`        Phase     ``thrust``, ``fuel_flow``
:class:`~cdadt.disciplines.mass.MassProperties`          Phase     ``weight``, ``fuel_burn_final``
======================================================== ========= ===========================

Swapping a discipline
---------------------

Both models list their disciplines as a class attribute, so replacing one is a subclass and
nothing else changes::

    class VLMPhaseModel(JetTransportPhaseModel):
        disciplines = (VLMAerodynamics, Propulsion, MassProperties)

    analysis = SizingAnalysis(aircraft, profile, phase_model=VLMPhaseModel)

:class:`~cdadt.model.Part25PhaseModel` is exactly this pattern, shipped: it replaces the
aerodynamics discipline with one that deploys takeoff flaps for the engine-out climb
condition, because 14 CFR 25.121(b) specifies them and OpenConcept's example does not. It is a
subclass, not a flag, so the choice is visible in the model the run was made with.

The sizing loop
---------------

:class:`~cdadt.model.SizingModel` closes the circularity that *is* aircraft sizing:

.. math::

   \mathrm{MTOW} = \mathrm{OEW}(\mathrm{MTOW}, \text{geometry})
                   + W_\text{payload}
                   + W_\text{fuel}(\mathrm{MTOW}, \text{mission})

A heavier aircraft burns more fuel and more fuel makes it heavier. One Newton solver, with
``solve_subsystems`` on, drives that residual to zero *at the same time* as the mission's own
implicit states -- the phase durations solved against altitude and range targets, the throttle
solved for zero acceleration, and the decision speed V\ :sub:`1` solved so that continuing and
aborting cover the same distance. Solving them in sequence instead would converge to a
different, wrong answer.

Which fuel closes the loop matters. It is the fuel burned by the end of **loiter**, the last
phase fuel accumulates through, not the block fuel at the end of descent. Closing on block
fuel would size an aircraft that carries no reserves -- and would converge just as readily,
which is what makes the mistake worth naming.

Design parameters
-----------------

:class:`~cdadt.aircraft.AircraftDefinition` publishes every parameter it holds as an
independent variable at the top of the model. Two consequences:

* Any of them can be an optimizer design variable without further plumbing.
* A quantity the model *computes* must not appear in the definition, because that would
  declare the same variable twice. Tail areas, MAC, wetted areas, OEW, MLW and MTOW are all
  absent from ``cases/b738_aircraft.yaml`` for this reason, and OpenMDAO rejects the model if
  one is added back.
