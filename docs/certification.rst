The certification basis
=======================

A certification-driven design is one where the requirements are not commentary on the answer
but constraints on the search, and where every one of them can be traced from a regulation,
through a number with a stated source, to the quantity that was actually evaluated. That chain
is what this part of cdadt exists to produce.

How a requirement is stated
---------------------------

.. code-block:: yaml

   requirements:
     - type: engine_out_climb_gradient
       limit: 0.024
       units: rad
       regulation: 14 CFR 25.121(b)(1)(i)
       source: Two-engine aeroplane, 2.4% second-segment minimum

``regulation`` and ``source`` are **required**, at construction as well as in the case file. The
limit itself is never a default: which runway, which aerodrome, which aeroplane class and which
engine count belong to the operating case, not to the code. A constraint that cannot name where
its number came from is a constraint nobody can defend, and this tool exists to produce reports
that get argued from.

Use ``regulation: design`` for a programme decision or a modelling limit rather than a rule, and
say which in ``source``. A design decision presented as a regulation is worse than no citation.

What each requirement class constrains
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 10 30 34

   * - ``type``
     - Sense
     - Evaluated on
     - Notes
   * - ``balanced_field_length``
     - upper
     - ``takeoff_field_length``
     - 14 CFR 25.113. The limit is the runway available, at the altitude and temperature
       intended.
   * - ``engine_out_climb_gradient``
     - lower
     - ``engine_out_climb_gradient``
     - 14 CFR 25.121(b): 2.4% two engines, 2.7% three, 3.0% four. See the warning below.
   * - ``throttle_limit``
     - upper
     - ``<phase>_throttle``
     - Not a regulation. The range the engine deck is fitted over. Requires a ``phase`` option.
   * - ``maximum_takeoff_weight``
     - upper
     - ``MTOW``
     - A type-certificate structural limit.
   * - ``design_range``
     - lower
     - ``mission_range_flown``
     - Normally satisfied by construction; see the note below before using it.
   * - ``response_limit``
     - either
     - any response
     - The escape hatch. Requires ``response`` and ``sense`` options.

Vector responses -- a throttle history has one value per node -- are one requirement, not
twenty-one. The whole history is constrained, and the node that governs is reported: the largest
for an upper limit, the smallest for a lower one.

Constraints are scaled by their own limit before being handed to the optimizer. Without that, a
climb gradient in hundredths of a radian is numerically invisible next to a field length in
thousands of feet.

.. warning::

   **The engine-out climb gradient is evaluated clean.** 14 CFR 25.121(b) specifies the
   second-segment gradient with the landing gear retracted *and the takeoff flaps set*. The
   black box evaluates its engine-out climb condition in the clean configuration, which
   produces less drag and therefore a higher gradient. The number this requirement constrains is
   consequently optimistic against the regulation as written.

   cdadt does not correct it. Correcting it would mean changing what the box computes, which is
   precisely what the black-box boundary exists to prevent. It is recorded here and in
   :doc:`validation` instead of being silently absorbed.

What cannot be constrained, and why
-----------------------------------

Every requirement is a function of quantities the box publishes. That is a real limit, and it
is the honest one: cdadt does not compute physics, so it cannot enforce a regulation whose
governing quantity the box does not produce.

The clearest example is **reference landing approach speed**, §25.125 and the approach-category
limits that go with it. It needs the reference stall speed at maximum landing weight in the
landing configuration. The box publishes ``ac|aero|Vstall_land``, but that is an *input* -- a
number the case file sets -- so constraining a speed derived from it would constrain a constant,
which is worse than not constraining it at all. Producing the real quantity would mean writing a
stall-speed model inside cdadt, which is the reimplementation the boundary exists to prevent.

Others in the same position, all listed in :doc:`validation`: minimum control speed on the
ground and in the air (§25.149), the landing climb and approach climb gradients (§25.119,
§25.121(c) and (d)), centre-of-gravity range and static margin, landing field length, and fuel
tank capacity.

Reading the result
------------------

Each evaluated requirement reports a margin defined so that its sign means the same thing
either way: ``limit - value`` for an upper limit, ``value - limit`` for a lower one. Positive is
compliant. Three statuses:

``MET``
    Satisfied with margin.

``ACTIVE``
    Satisfied and sitting on the limit, within 0.01% of it. An active requirement is one that
    shaped the design, and that is the most interesting row in the table.

``VIOLATED``
    Not satisfied. An optimization that ends with any violated requirement is reported as not
    having succeeded, whatever the driver said.

The traceability matrix
-----------------------

The artefact the module exists to produce. Every requirement, the regulation behind it, the
number and where that number came from, the value achieved, the margin and the status:

.. code-block:: text

   regulation               requirement                                          value       limit      margin  units  status
   ------------------------------------------------------------------------------------------------------------------------
   14 CFR 25.113            Balanced field length within the runway available  6263.7147  8000.0000  1736.2853  ft     MET
   14 CFR 25.121(b)(1)(i)   OEI second-segment climb gradient                     0.0554     0.0240     0.0314  rad    MET
   design                   Throttle within the engine deck's range in climb      1.0000     1.0000    -0.0000  -      ACTIVE
   design                   Throttle within the engine deck's range in cruise     0.8287     1.0000     0.1713  -      MET

   Where each limit came from
   --------------------------
     balanced_field_length (14 CFR 25.113): Design field length, 8000 ft dry runway at sea level, ISA
     engine_out_climb_gradient (14 CFR 25.121(b)(1)(i)): Two-engine aeroplane, 2.4% second-segment minimum
     throttle_limit_climb (design): The CFM56 surrogate inside the black box is not fitted above throttle 1
     throttle_limit_cruise (design): The CFM56 surrogate inside the black box is not fitted above throttle 1

   4 of 4 requirements met, 1 active, 0 violated.

``cdadt optimize <case> --json out.json`` writes the same information as structured data, one
record per requirement, so a study is archivable without re-running it.

Adding a requirement
--------------------

Subclass :class:`~cdadt.certification.Requirement`, then include it in a
:class:`~cdadt.certification.RequirementCatalog` so a case file can name it:

.. code-block:: python

   from typing import ClassVar

   from cdadt import Requirement


   class ReserveFuelFraction(Requirement):
       """Fuel with reserves must exceed block fuel by a stated fraction."""

       kind: ClassVar[str] = "reserve_fuel_fraction"
       response: ClassVar[str] = "total_fuel"
       sense: ClassVar[str] = "lower"
       title: ClassVar[str] = "Fuel with reserves"

.. code-block:: python

   from cdadt import Optimizer, RequirementCatalog, SHIPPED_REQUIREMENTS

   catalog = RequirementCatalog([*SHIPPED_REQUIREMENTS, ReserveFuelFraction])
   optimizer = Optimizer(analysis, requirements=catalog)

Defining the class is not enough on its own, and that is deliberate. An earlier design registered
subclasses automatically into a class-level dictionary, which made the set of available
requirements depend on what had been imported and let one study's classes leak into another's.
The catalogue is instance state: a study owns its own, and two are independent. See
:doc:`architecture`.

The class fixes the physics -- which response tests the requirement and which way the inequality
runs -- and leaves the number, its regulation and its source to the case file. That division is
the point: the named classes cannot get the sense or the response wrong, which the generic
``response_limit`` can.

If the quantity you need is not one the box publishes, that is not a gap in cdadt to be filled
by adding a calculation here. It is a statement about the black box, and it belongs in
:doc:`validation`.
