.. _architecture:

************
Architecture
************

cdadt is built from four abstractions. This page states what each one is responsible for
and, where a rule looks like a stylistic preference, what failure it exists to prevent.

Variables
=========

A :class:`~cdadt.core.variables.Variable` is a declaration: a name, its units, and whether
it carries one value per analysis node. Names follow the OpenConcept convention -- a
pipe-separated path such as ``ac|geom|wing|S_ref`` or ``fltcond|CL`` -- so a cdadt
declaration can be compared directly against the promoted names of a built OpenConcept
model.

Units are validated against OpenMDAO's own unit library at declaration time. cdadt wraps
that check in :func:`~cdadt.core.variables.is_valid_units` because OpenMDAO's
``valid_units`` evaluates the string as a Python expression and therefore raises
``SyntaxError`` on malformed input such as ``"m**"`` rather than returning ``False``.

A :class:`~cdadt.core.variables.VariableSet` is an immutable, name-keyed collection with
set algebra. Union **raises** on two declarations that share a name but disagree on units,
rather than picking one:

.. code-block:: python

   metric = VariableSet([Variable("thrust", "N", vectorized=True)])
   imperial = VariableSet([Variable("thrust", "lbf", vectorized=True)])
   metric | imperial          # VariableConflictError

Choosing a winner here would put a factor of 4.448 into the model and leave no trace of
where it came from.

Configuration
=============

An :class:`~cdadt.core.configuration.AircraftConfiguration` holds every number that
describes a design and every constant a method needs. It is immutable;
:meth:`~cdadt.core.configuration.AircraftConfiguration.with_overrides` returns a new
configuration rather than mutating the receiver.

**There is no way to read a value with a fallback.**
:meth:`~cdadt.core.configuration.AircraftConfiguration.value` has no ``default``
parameter, and it will not acquire one. Requesting an unconfigured name raises
:class:`~cdadt.core.configuration.MissingConfigurationError`.

This is the single most consequential rule in the framework. The tempting alternative is to
give a method constant a default drawn from a textbook -- ``k = 1.23`` from §25.125, say --
on the grounds that the value is sourced and correct. But a sourced number used as a
default is still a hardcoded constant: it applies itself to configurations that never
mentioned it, and a result computed from it is a result nobody chose. Sourcing a number
tells you it is defensible for the case in the source; it does not tell you it is
defensible for the case being run.

Providers therefore call
:meth:`~cdadt.core.configuration.AircraftConfiguration.require_all` in
:meth:`~cdadt.core.provider.Provider.validate_configuration`, which reports *every* absent
constant at once. Reporting them one at a time is how a configuration acquires a value that
was guessed to make a traceback go away.

Configuration leaves may carry a ``source`` string alongside ``value`` and ``units``. That
string is what the certification traceability report cites, so a reader can check any
number against its reference.

Providers
=========

A :class:`~cdadt.core.provider.Provider` is *how* something is computed. It declares the
variables it consumes and produces, cites the method it implements, and builds the OpenMDAO
subsystems that produce those variables.

Providers hold no numeric literals. Every constant comes from the configuration, which is
what makes a provider's :attr:`~cdadt.core.provider.Provider.reference` citation meaningful:
the citation describes the method, and the configuration records the numbers the method was
run with.

Disciplines
===========

A :class:`~cdadt.core.discipline.Discipline` is one engineering domain -- aerodynamics,
propulsion, weights, geometry, stability -- as an object with its own state. It delegates
physics to a provider, so an empirical buildup and a vortex-lattice analysis satisfy the
same interface and the surrounding model does not know which is in use.

Two rules govern disciplines:

**A discipline owns only itself.** :meth:`~cdadt.core.discipline.Discipline.build` adds
subsystems to the group it is given and sets up its own promotions. It never reaches into a
sibling, never inspects its parent, and never modifies anything it did not create. Coupling
happens by matching promoted names.

**No global state.** Everything a discipline needs arrives through its constructor. There is
no registry, no module-level cache, and no import-time side effect. This is not tidiness:
every sizing iteration and every optimizer function evaluation builds aircraft models, and
shared mutable state between them would make results depend on evaluation order.

Coupling is checked before anything is built
============================================

:func:`~cdadt.core.discipline.check_coupling` takes the discipline set and the variables the
mission supplies externally, and rejects two situations:

* a discipline requires a variable that nothing provides and nothing supplies externally,
  and
* two disciplines both provide the same variable.

Neither is caught by OpenMDAO. An unconnected promoted input is legal there -- it simply
holds its declared default -- so the first situation produces a converged model that
computed its answer from a number nobody set. The second makes the result depend on the
order subsystems were added.

.. code-block:: python

   check_coupling([aero, propulsion, weights],
                  externally_supplied=[Variable("fltcond|q", "N*m**-2", vectorized=True),
                                       Variable("throttle", None, vectorized=True)])

Every problem in the set is reported at once, for the same reason ``require_all`` reports
every missing constant at once.
