The certification basis
=======================

A certification-driven design is one where the requirements are not commentary on the answer
but constraints on the search, and where every one of them can be traced from a regulation,
through a number with a stated source, to the quantity that was actually evaluated. That chain
is what this part of cdadt exists to produce.

How a constraint is stated
--------------------------

A constraint is one ``prob.model.add_constraint(...)`` call -- exactly what OpenConcept's own
optimizing examples write -- plus two fields those examples have nowhere to put:

.. code-block:: yaml

   constraints:
     - name: engine_out_climb_gradient
       lower: 0.024
       units: rad
       regulation: 14 CFR 25.121(b)(1)(i)
       source: Two-engine aeroplane, 2.4% second-segment minimum
       title: OEI second-segment climb gradient

     - name: climb_throttle
       lower: 0.01
       upper: 1.05
       title: Climb throttle within the engine deck

Both are legal, and they are different kinds of thing. A constraint that names a regulation
*and* a source is certification evidence and appears in the traceability matrix as such. One
that names neither is a design constraint, and the matrix says so rather than implying every row
is defensible against a rule.

Neither field is ever supplied by default. Which runway, which aerodrome, which aeroplane class
and which engine count belong to the operating case, not to the code. Use ``regulation: design``
for a programme decision or a modelling limit rather than a rule, and say which in ``source``: a
design decision presented as a regulation is worse than no citation.

What may be constrained
-----------------------

Anything the black box publishes. A constraint's ``name`` is either one of the response names
the disciplines report -- the full list is in :doc:`interface` -- or a raw path inside the box.
The ones a certification argument is normally built from:

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - ``name``
     - Sense
     - Notes
   * - ``takeoff_field_length``
     - ``upper``
     - 14 CFR 25.113. The limit is the runway available, at the altitude and temperature
       intended.
   * - ``engine_out_climb_gradient``
     - ``lower``
     - 14 CFR 25.121(b): 2.4% two engines, 2.7% three, 3.0% four. See the warning below.
   * - ``MTOW``
     - ``upper``
     - A type-certificate structural limit.
   * - ``mission_range_flown``
     - ``lower``
     - Normally satisfied by construction: the mission is flown to the range the case asks for.
   * - ``climb_throttle``, ``cruise_throttle``, ``descent_throttle``
     - band
     - Not a regulation. The range the engine deck is fitted over. Written as a two-sided bound,
       exactly as ``B738_aerostructural.py`` writes it.

Every form ``add_constraint`` accepts is expressible: one-sided, two-sided, equality, with any of
OpenMDAO's ``indices``, ``linear`` and scaling arguments. Nothing is interpreted on the way
through -- see :doc:`configuration` for the full key list.

Vector responses -- a throttle history has one value per node -- are one constraint, not
twenty-one. The whole history is constrained, and the node that governs is reported: for a band,
the node of least margin on either side; for a one-sided bound, the extreme value.

Constraints are scaled by the magnitude of their own bound before being handed to the optimizer.
Without that, a climb gradient in hundredths of a radian is numerically invisible next to a field
length in thousands of feet.

.. warning::

   **The engine-out climb gradient is evaluated clean.** 14 CFR 25.121(b) specifies the
   second-segment gradient with the landing gear retracted *and the takeoff flaps set*. The
   black box evaluates its engine-out climb condition in the clean configuration, which
   produces less drag and therefore a higher gradient. The number this constraint bounds is
   consequently optimistic against the regulation as written.

   cdadt does not correct it. Correcting it would mean changing what the box computes, which is
   precisely what the black-box boundary exists to prevent. It is recorded here and in
   :doc:`validation` instead of being silently absorbed.

What cannot be constrained, and why
-----------------------------------

Every constraint is a bound on a quantity the box publishes. That is a real limit, and it is the
honest one: cdadt does not compute physics, so it cannot enforce a regulation whose governing
quantity the box does not produce.

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

Each evaluated constraint reports a margin defined so that its sign means the same thing whatever
form the bound takes: ``upper - value`` for an upper limit, ``value - lower`` for a lower one,
whichever is smaller for a band, and ``-|value - equals|`` for an equality. Positive is compliant.
Three statuses:

``MET``
    Satisfied with margin.

``ACTIVE``
    Satisfied and sitting on the bound, within 0.01% of it. An active constraint is one that
    shaped the design, and that is the most interesting row in the table.

``VIOLATED``
    Not satisfied. An optimization that ends with any violated constraint is reported as not
    having succeeded, whatever the driver said.

The traceability matrix
-----------------------

The artefact the module exists to produce. Every constraint, the regulation behind it, the number
and where that number came from, the value achieved, the margin and the status. The two kinds are
separated at the bottom, so a reader is never left inferring which rows are evidence:

.. code-block:: text

   regulation              constraint                                           value          bound        margin  units  status
   -------------------------------------------------------------------------------------------------------------------------------
   14 CFR 25.113           Balanced field length within the runway available  6263.7147   <= 8000.0000  1736.2853   ft     MET
   14 CFR 25.121(b)(1)(i)  OEI second-segment climb gradient                     0.0554     >= 0.0240     0.0314   rad    MET
   -                       Climb throttle within the engine deck                 1.0500  0.0100..1.0500    -0.0000  -      ACTIVE
   -                       Cruise throttle within the engine deck                0.8287  0.0100..1.0500     0.2213  -      MET

   Where each limit came from
   --------------------------
     takeoff_field_length (14 CFR 25.113): Design field length, 8000 ft dry runway at sea level, ISA
     engine_out_climb_gradient (14 CFR 25.121(b)(1)(i)): Two-engine aeroplane, 2.4% second-segment minimum

   Design constraints with no stated regulation or source: climb_throttle, cruise_throttle.
   These bound the design; they are not certification evidence.

   4 of 4 constraints met, 1 active, 0 violated.

``cdadt optimize <case> --json out.json`` writes the same information as structured data, one
record per constraint -- including its ``traceable`` flag -- so a study is archivable without
re-running it.

The classes behind it
---------------------

Three, and the division between them is the module's whole design:

:class:`~cdadt.certification.Constraint`
    One constraint: what it bounds, where that quantity lives in the box, its provenance, and
    how to register it on a model and evaluate it against a design.

:class:`~cdadt.certification.ConstraintResult`
    That constraint evaluated against one converged design: the governing value, the margin, and
    the status. It owns ``ACTIVE_TOLERANCE``, which is what "sitting on the bound" means.

:class:`~cdadt.certification.CertificationBasis`
    The set a design is held to. Refuses duplicate names, registers the whole set on a model,
    evaluates the whole set against a design, and renders the matrix.

Nothing here knows what a field length *is*. A constraint is a bound on a named quantity plus its
provenance, and the physics belongs to the black box. That is why adding a new constraint takes
no Python at all -- it is six lines of YAML -- and why cdadt cannot get the sense of a regulation
wrong in code that a case file would then be unable to correct.

If the quantity you need is not one the box publishes, that is not a gap in cdadt to be filled by
adding a calculation here. It is a statement about the black box, and it belongs in
:doc:`validation`.
