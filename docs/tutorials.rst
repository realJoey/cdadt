Tutorials
=========

Five things a study actually needs to do, in increasing order of how much code they take. The
first three take none.

1. Change the aircraft
----------------------

Copy ``cases/b738.yaml``, edit the numbers, run it. Everything in the ``aircraft`` section is a
design parameter, and the black box is a *sizing* model -- empirical weight and drag buildups
and a rubberized engine -- so a changed parameter produces a consistent clean-sheet design
rather than an inconsistent 737.

.. code-block:: yaml

   aircraft:
     ac|geom|wing|AR:
       value: 11.5                       # was 9.45
       source: Design study, higher aspect ratio wing
     ac|propulsion|engine|rating:
       value: 24.0e3                      # was 27.0e3
       units: lbf
       source: Design study, reduced thrust

.. code-block:: bash

   cdadt size cases/my_aircraft.yaml

Keep the ``source`` field truthful as you edit. It is what separates a design decision from a
leftover.

If a parameter name is wrong, cdadt says so before anything runs, and suggests the nearest
match. ``cdadt inspect`` lists all 34.

2. Change the mission
---------------------

.. code-block:: yaml

   mission:
     parameters:
       mission_range: {value: 3500, units: nmi}     # was 2800
       cruise|h0:     {value: 37000, units: ft}     # was 35000

Watch the continuation ladder when you make the mission substantially harder. Its job is to walk
the solver from something easy to the design mission, and the shipped ladder was built for a
2800 nmi mission at FL350. If a run fails to converge, add a rung:

.. code-block:: yaml

     continuation:
       - description: short range at low altitude, gentle descent
         parameters:
           mission_range: {value: 500, units: nmi}
           cruise|h0:     {value: 5000, units: ft}
           reserve_range: {value: 100, units: nmi}
           reserve|h0:    {value: 1000, units: ft}
         schedule:
           descent: {Ueas: {value: [252, 250], units: kn}, vs: {value: -800, units: ft/min}}

       - description: half range at intermediate altitude          # <- new rung
         parameters:
           mission_range: {value: 1800, units: nmi}
           cruise|h0:     {value: 25000, units: ft}
         schedule:
           descent: {Ueas: {value: [252, 250], units: kn}, vs: {value: -800, units: ft/min}}

       - description: design range and altitude, gentle descent
         schedule:
           descent: {Ueas: {value: [252, 250], units: kn}, vs: {value: -800, units: ft/min}}

Run with ``-v`` to watch each rung as it converges. See :doc:`mission`.

3. Ask a different question of the same aeroplane
-------------------------------------------------

Two studies of one aircraft differ only in their ``optimization`` section. To minimize maximum
takeoff weight instead of fuel, over span alone, subject to the field length:

.. code-block:: yaml

   optimization:
     driver: {name: IPOPT, maxiter: 40, tol: 1.0e-6, derivative_mode: fwd}
     objective: {name: MTOW, units: kg, sense: minimize, ref: 8.0e4}
     design_variables:
       - {name: ac|geom|wing|AR, lower: 7.0, upper: 13.0}
     requirements:
       - type: balanced_field_length
         limit: 7000.0
         units: ft
         regulation: 14 CFR 25.113
         source: Shorter runway, 7000 ft dry at sea level, ISA

Nothing else changes, and nothing in Python changes. See :doc:`optimization`.

4. Add a certification requirement
----------------------------------

If the quantity is already reported, the case file is enough -- use the generic
``response_limit``:

.. code-block:: yaml

     requirements:
       - type: response_limit
         response: total_fuel
         sense: upper
         limit: 20000.0
         units: kg
         regulation: design
         source: Usable fuel volume of the wing box as laid out

For a requirement you will state repeatedly, give it a class so the sense and the response
cannot be got wrong:

.. code-block:: python

   from typing import ClassVar

   from cdadt import Requirement


   class MaximumLandingWeight(Requirement):
       """Landing weight must not exceed the certificated structural limit."""

       kind: ClassVar[str] = "maximum_landing_weight"
       response: ClassVar[str] = "MLW"
       sense: ClassVar[str] = "upper"
       title: ClassVar[str] = "Maximum landing weight within the structural limit"

Then hand the optimizer a catalogue that includes it, and ``type: maximum_landing_weight`` works
in a case file:

.. code-block:: python

   from cdadt import Optimizer, RequirementCatalog, SHIPPED_REQUIREMENTS

   optimizer = Optimizer(
       analysis,
       requirements=RequirementCatalog([*SHIPPED_REQUIREMENTS, MaximumLandingWeight]),
   )

The class fixes the physics and leaves the number, its regulation and its source to the case.
The catalogue step is deliberate rather than automatic: registering subclasses on definition
would be shared mutable state, which the package does not have and the suite forbids. See
:doc:`architecture` and :doc:`certification`.

If the quantity you need is *not* something the black box publishes, stop. Adding a calculation
to cdadt to produce it is exactly what the boundary exists to prevent; the gap belongs in
:doc:`validation`.

5. Add a discipline
-------------------

.. code-block:: python

   from typing import ClassVar

   from cdadt import Aircraft, Config, Discipline, Response, SizingAnalysis
   from cdadt.disciplines import AIRCRAFT_DISCIPLINES


   class Cost(Discipline):
       """Acquisition and operating cost, as far as the box reports it."""

       discipline_name: ClassVar[str] = "cost"
       description: ClassVar[str] = "Acquisition and operating cost"
       owned_patterns: ClassVar[tuple[str, ...]] = ("ac|cost|*",)
       reported: ClassVar[tuple[Response, ...]] = (
           Response("acquisition_cost", "costs.acquisition", "USD", optional=True),
       )

Declare what it owns and what it reports; that is the whole interface. The ownership check will
tell you immediately if the patterns overlap an existing discipline's, and
:meth:`~cdadt.disciplines.base.Discipline.collect` will tell you if the box does not publish
what the discipline claims to report.

Remember what a discipline is and is not: it owns the *interface* to its domain, not the
physics. If your new class starts computing something, it is in the wrong repository.

6. Drive a different black box
------------------------------

cdadt is not tied to the B738 case. Any OpenConcept group that takes ``num_nodes`` and exposes
its design parameters as independent variables can be named in a case file:

.. code-block:: yaml

   black_box:
     model: my_package.my_sizing:MySizingAnalysis
     num_nodes: 11

Then run ``cdadt inspect`` against it first. Whatever it publishes is what cdadt can set,
constrain and report; the ownership patterns of the existing disciplines will route the
``ac|``-named ones, and anything else needs a discipline of its own -- which the ownership check
will point out by name.
