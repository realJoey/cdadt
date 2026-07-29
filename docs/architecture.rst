Architecture
============

Every piece of cdadt is a class with encapsulated state. There is no global state, no
module-level cache, and no free function that does discipline work. This page says what the
classes are, what each owns, and why the division is drawn where it is.

The layers
----------

.. code-block:: text

   CommandLineInterface      composes Command subclasses; owns the parser and the dispatch
     |                         SizeCommand / OptimizeCommand are StudyCommands
     |                         InspectCommand reads the interface without running
     |
   Config                    one YAML case file, validated hard
     |                         read through CaseFileSection, which carries its own address
     |
     +-- Aircraft            composes the airframe disciplines, routes every ac| name
     |     +-- Geometry      wing, fuselage, nacelles, gear
     |     +-- Aerodynamics  span efficiency, airfoil, flaps, envelope
     |     +-- Propulsion    engine rating and count
     |     +-- Stability     empennage shape, tail areas
     |     +-- Structures    load-carrying airframe mass breakdown
     |     +-- Weights       payload, cabin, the closed weight rollup
     |
     +-- Performance         owns the InitialConditions and the ContinuationLadder
     |
     +-- OpenConceptSizingBox   the black box: loaded by name, set, converged, read
           |                      RunDirectory decides where its problem writes
   SizingAnalysis            builds, converges, reads -> SizingResults
     |                         the coordinator; every collaborator is injectable
   Optimizer                 declares design variables, objective and CertificationBasis,
     |                       converges a baseline, drives -> OptimizationOutcome
     |
   StudyArtifacts            writes the record into the run directory
         MissionTrajectory   the seven steady phases, discovered and ordered
         TakeoffTrajectory   the balanced field: the continued and rejected paths

The command line is at the top rather than off to one side because it is the only caller that
supplies a :class:`~cdadt.blackbox.RunDirectory`, and therefore the only one that causes anything
to be written to disk. A study driven from Python writes nothing unless it asks to. See
:doc:`artifacts`.

The arrangement is open/closed in three places, each tested rather than asserted: a new
discipline is a new :class:`~cdadt.disciplines.base.Discipline` passed to ``Aircraft``, a new
command is a new :class:`~cdadt.cli.Command` passed to ``CommandLineInterface``, and a new black
box is a different ``model:`` line in the case file. None of the three requires editing anything
that already exists.

What a discipline is
--------------------

A discipline is one engineering domain, modelled as a class, and it owns exactly two things:

**The parameters it puts into the box.** Which ``ac|`` variables belong to its domain, their
values, their units and their provenance. Held as :class:`~cdadt.parameters.Parameter` objects,
reachable only through methods that validate what they are given -- a non-finite value is
rejected at the point of assignment rather than surfacing later as an unexplained convergence
failure.

**The responses it reads back out.** Which quantities the box produces belong to its domain,
where they live inside it, and in what units to read them. Held as
:class:`~cdadt.parameters.Response` objects, which is what allows the whole interface to be
listed without running anything.

What a discipline is **not**
----------------------------

It does not compute physics. There is no ``compute``, no residual and no partial derivative in
any of them, because the aerodynamics, the weight correlations, the engine deck, the tail
sizing and the entire mission are computed inside the black box by OpenConcept. A cdadt
discipline is the encapsulated, validated interface to its slice of that box.

This is stated bluntly because the alternative is worse. A ``cdadt.disciplines.aerodynamics``
module that *looked* like it modelled aerodynamics, in a tool whose results are quoted in a
thesis, would be a page of documentation away from a false claim. It sets the inputs to somebody
else's drag buildup and reads two lift coefficients back.

:class:`~cdadt.disciplines.base.Discipline` is an abstract base, and ``discipline_name`` is
genuinely abstract -- it is the one thing no domain can inherit, being the key it is reported and
looked up under. A subclass that omits it is refused when the class is *written*, by
``__init_subclass__``, rather than surfacing later as a duplicate-name error from ``Aircraft`` or
not at all. Nothing else is abstract, because a subclass overrides no methods: it declares three
class attributes and that is the whole contract.

Certification is a domain, and is not a discipline
---------------------------------------------------

:class:`~cdadt.certification.CertificationBasis` encapsulates the certification domain, and
:class:`~cdadt.analysis.SizingAnalysis` owns it alongside the disciplines. It is deliberately
**not** a :class:`~cdadt.disciplines.base.Discipline`, and the reason is the definition above: a
discipline owns a slice of the black box's *variable interface* -- variables it sets, responses
it reports. Certification sets nothing and publishes nothing. It reads quantities the other
disciplines already report and judges them against stated limits, which is a different
relationship to the box, and forcing it into the same base class would describe it wrongly: it
would appear in the ownership map owning nothing and in the results table reporting nothing.

It is owned by the analysis rather than by :class:`~cdadt.optimization.Optimizer`, which used to
build it. That mattered in practice: asking whether a *sized* aeroplane meets its certification
basis required constructing an optimizer that was never going to run. A sizing study can now
answer it directly, and a test does exactly that.

Ownership, and why it is checked
--------------------------------

Each discipline declares the variables it owns as shell-style patterns:

===============  ==================================================================
Geometry         ``ac|geom|wing|*``, ``ac|geom|fuselage|*``, ``ac|geom|nacelle|*``,
                 ``ac|geom|maingear|*``, ``ac|geom|nosegear|*``
Aerodynamics     ``ac|aero|*``
Propulsion       ``ac|propulsion|*``
Stability        ``ac|geom|hstab|*``, ``ac|geom|vstab|*``
Structures       nothing
Weights          ``ac|weights|*``, ``ac|num_*``, ``ac|cabin_pressure``
Performance      ``mission.*``
===============  ==================================================================

:class:`~cdadt.aircraft.Aircraft` routes each declared parameter to the one discipline that
claims it, and refuses a parameter that no discipline owns or that two would claim. A contract
test then walks the *whole settable set of a built box* -- 241 variables for the shipped case,
read off the live model rather than from a list -- and asserts that every one of them has
exactly one owner. A variable cdadt could set but no discipline claims is a hole in the model
of the interface, whether or not any case file happens to set it.

Two of those rows deserve explanation.

**Structures owns nothing, and that is the honest answer.** Primary structure is sized inside
the box by correlations that read geometry, maximum takeoff weight and maximum landing weight --
all owned elsewhere or computed by the box. There is no structural parameter left for cdadt to
set: no material, no load factor, no spar layout. Inventing one so that the class had state
would be a parameter nothing reads. What it does own is the *reporting* of structure: the seven
component masses, so that a weight figure can be argued with rather than only quoted.

**The empennage is Stability, not Geometry.** Its areas are outputs, not inputs -- the box sizes
both surfaces by tail volume coefficient -- so the shape parameters and the resulting areas
belong together, in the domain the method belongs to.

Performance is the odd one
--------------------------

:class:`~cdadt.disciplines.performance.Performance` is the one discipline whose encapsulated
state is not a bag of scalars. It owns the pair OpenConcept's own run scripts write by hand: the
:class:`~cdadt.mission.InitialConditions` -- everything ``set_values(prob, num_nodes)`` writes,
which is the design range, the cruise altitude, the reserve mission, the per-phase schedules and
the solver's starting guesses -- and the :class:`~cdadt.mission.ContinuationLadder` that makes
the hard ones reachable. Those are design inputs in exactly the sense the geometry parameters
are: change the range and you change the aeroplane. See :doc:`mission`.

It is also the discipline the certification constraints are written against: field length,
decision and safety speeds, engine-out climb gradient, fuel, and the throttle history of every
phase.

Adding a discipline
-------------------

Subclass :class:`~cdadt.disciplines.base.Discipline`, declare what it owns and what it reports,
and pass it to :class:`~cdadt.aircraft.Aircraft`:

.. code-block:: python

   from cdadt import Aircraft, Discipline, Response
   from cdadt.disciplines import AIRCRAFT_DISCIPLINES


   class Cost(Discipline):
       """Acquisition and operating cost, if the box publishes any."""

       discipline_name = "cost"
       description = "Acquisition and operating cost"
       owned_patterns = ("ac|cost|*",)
       reported = (Response("acquisition_cost", "costs.acquisition", "USD", optional=True),)


   aircraft = Aircraft(parameters, disciplines=(*AIRCRAFT_DISCIPLINES, Cost))

Nothing else changes. The ownership check will immediately tell you if the new patterns overlap
an existing discipline's, and :meth:`~cdadt.disciplines.base.Discipline.collect` will tell you
if the box does not publish what the new discipline claims to report.

The rules, and the tests that enforce them
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Rule
     - Test that enforces it
   * - No cdadt module imports OpenConcept
     - ``test_no_cdadt_module_imports_openconcept``
   * - No cdadt class inherits from OpenConcept
     - ``test_no_cdadt_class_inherits_from_openconcept``
   * - The OpenConcept working tree is untouched
     - ``test_the_openconcept_working_tree_is_clean``
   * - No local OpenConcept commit affects a module cdadt loads
     - ``test_no_locally_committed_openconcept_change_touches_a_module_cdadt_loads``
   * - Every settable variable of the box has exactly one owner
     - ``test_every_settable_variable_of_the_box_has_exactly_one_owner``
   * - Every required response exists in the box
     - ``test_every_reported_response_exists_in_the_box``
   * - The base ``Discipline`` cannot be instantiated
     - ``test_the_base_class_cannot_be_instantiated``

Design decisions worth knowing
------------------------------

**Values are validated where they are set, not where they are used.** A
:class:`~cdadt.parameters.Parameter` refuses a non-numeric or non-finite value in its setter. A
``NaN`` that reaches OpenMDAO propagates silently through a Newton solve and emerges as a
convergence failure with no attribution.

**Case files reject unknown keys.** A silently ignored key is the failure mode a configuration
file is most prone to: the run succeeds and answers a different question than the one that was
asked.

**Optimizations are checked against a cheap probe of the box before they start.** A small build
costs a fraction of a second and knows every name the box publishes, so a misspelled design
variable is a message with suggestions rather than an OpenMDAO error thrown out of ``setup``.

**Results are read whole.** Every discipline's responses are collected on every run, not
whichever few a caller asked for. That is what makes a run report complete and two runs
comparable without re-running either.

There is no shared mutable state
--------------------------------

Not as an aspiration -- as a property the suite enforces. Two contract tests parse and walk the
whole package:

``test_cdadt_has_no_module_level_mutable_state``
    No module binds a mutable object at import time. ``__all__`` is exempt: it is a list by
    language convention, is never mutated, and describes the module rather than holding state.

``test_cdadt_has_no_class_level_mutable_state``
    No class carries a mutable class attribute. Immutable class attributes -- the ownership
    patterns, the response tuples, the allowed-key tuples on every config section -- are the
    intended way to declare what a class *is*, and are unaffected.

Everything a discipline, an aircraft, a mission or an analysis holds is instance state, reached
through properties that validate what they are given.

**This cost two design changes, which is the point of testing it.** An earlier version gave a
``Requirement`` base class a class-level ``registry`` dictionary, populated by
``__init_subclass__``, so that declaring a subclass made its ``kind`` nameable in a case file.
Convenient, and shared mutable state: two studies in one process shared one registry, defining a
class anywhere mutated it, and what a case file resolved to depended on what happened to have
been imported.

The fix went further than moving the mapping onto an instance. There is no requirement class
hierarchy at all now. A constraint is a bound on a named quantity plus its provenance --
:class:`~cdadt.certification.Constraint` -- and the set a design is held to is a
:class:`~cdadt.certification.CertificationBasis` built from the case file's own ``constraints``
list. Nothing has to be registered, subclassed or imported to make a new one nameable, because
there is nothing to name: the quantity is a response the disciplines already report, and the
regulation is a string in the YAML.

.. code-block:: python

   from cdadt import CertificationBasis, Constraint

   basis = CertificationBasis.from_specs(config.constraints, catalog)

Two bases are independent, and a duplicate constraint name is refused by name rather than
resolved by letting the later one win -- which is what OpenMDAO would otherwise do silently.

The only module-level *functions* in the package are argument parsing in :mod:`cdadt.cli` and
case-file validation helpers in :mod:`cdadt.config`. Neither performs a discipline calculation --
there are none to perform in cdadt, because the physics is inside the black box.
