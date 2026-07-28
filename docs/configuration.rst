The case file
=============

A cdadt study is defined by one YAML file, not by a script. The file names the black box, the
aircraft, the mission, the solver, and -- for an optimization -- the design variables, the
objective and the certification requirements.

Every key is documented below. Every section rejects keys it does not recognise, because a
silently ignored key in a configuration file means the run succeeds and answers a different
question than the one that was asked.

Skeleton
--------

.. code-block:: yaml

   black_box:        # required
   solver:           # optional
   aircraft:         # required
   initial_guesses:  # optional
   mission:          # required
   optimization:     # optional; present makes it an optimization study

``black_box``
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Required
     - Meaning
   * - ``model``
     - yes
     - The sizing analysis to drive, as ``module.path:ClassName``. Loaded by name at run time;
       never imported by cdadt, never modified, never subclassed.
   * - ``num_nodes``
     - yes
     - Analysis points per mission phase. **Must be odd**: the box integrates fuel burn with
       Simpson's rule, which needs 2N + 1 points. 21 for the shipped sizing case; 11 for the
       optimization, where the grid is traded against iteration count.

``solver``
----------

Configures the Newton solve that closes the weight loop and the mission's own balances
together. All keys optional; the defaults are OpenConcept's own sizing run script's.

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Key
     - Default
     - Meaning
   * - ``maxiter``
     - 20
     - Newton iteration limit. The shipped optimization case raises it to 50: an optimizer
       visits designs a human would not.
   * - ``atol``, ``rtol``
     - 1e-9
     - Absolute and relative residual tolerances.
   * - ``iprint``
     - -1
     - Solver print level. ``2`` prints every iteration.
   * - ``err_on_non_converge``
     - ``true``
     - Whether a failed solve raises. Leave it true. A non-converged mission still produces
       numbers -- a negative field length, a range that misses the one requested -- and nothing
       about them says so.

``aircraft``
------------

One entry per design parameter, keyed by the name the black box publishes:

.. code-block:: yaml

   aircraft:
     ac|geom|wing|S_ref:
       value: 124.6
       units: m**2
       source: b737.org.uk technical specifications

.. list-table::
   :header-rows: 1
   :widths: 16 14 70

   * - Key
     - Required
     - Meaning
   * - ``value``
     - yes
     - The number, in ``units``. Must be finite.
   * - ``units``
     - no
     - OpenMDAO unit string (``m**2``, ``lbf``, ``kn``, ``deg``). Omit for a dimensionless
       quantity such as aspect ratio.
   * - ``source``
     - no, but
     - Where the number came from. Technically optional; in practice required, and the shipped
       cases have one for all 34 parameters. A number with no provenance is indistinguishable
       from a guess in the report that quotes it.

Every name here must be an **independent variable** of the box. Quantities the box computes --
``ac|weights|OEW``, ``ac|geom|hstab|S_ref``, ``ac|aero|CLmax_TO`` -- are outputs, and setting
one would be overwritten by the next solve. cdadt checks the whole list against the built model
and names every offender at once, with suggestions. ``cdadt inspect <case>`` lists what is
available.

``initial_guesses``
-------------------

Same layout as ``aircraft``, but held to a looser standard, on purpose:

.. code-block:: yaml

   initial_guesses:
     ac|weights|MTOW: {value: 50.0e3, units: kg}

Maximum takeoff weight is an *output* -- it is what the weight closure solves for -- so it can
never be a design parameter or a design variable. Writing a value onto it before the first solve
is nonetheless meaningful: it is where Newton starts, and on a sizing loop that is frequently
the difference between converging and not. These decide whether the solver converges, not what
it converges to.

``mission``
-----------

See :doc:`mission` for what the schedules and the continuation ladder mean. The keys:

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Key
     - Required
     - Meaning
   * - ``parameters``
     - yes
     - Mission-level values, as ``{value, units}`` entries: ``mission_range``,
       ``reserve_range``, ``cruise|h0``, ``reserve|h0``, ``loiter|h0``, ``loiter_duration``,
       ``takeoff|h``. Anything omitted keeps the box's own default.
   * - ``schedule``
     - yes
     - One entry per steady-flight phase, each giving both ``Ueas`` and ``vs``. **All seven
       phases are required**, and both keys are required in each: an unscheduled phase flies
       its component's placeholder values, and setting one of the pair alone flies a profile
       that is half inherited.
   * - ``takeoff_speed_guess``
     - no
     - True-airspeed seed for the three ground-roll phases. Default 100 kn.
   * - ``continuation``
     - no
     - A list of progressively harder missions, converged in order before the design mission.
   * - ``mission_path``
     - no
     - Name of the mission subsystem inside the box. Default ``mission``.

A schedule value may be a single number (constant), two numbers (linearly interpolated across
the phase), or exactly ``num_nodes`` numbers. Any other length is an error rather than being
broadcast, because broadcasting would quietly fly a different mission.

.. code-block:: yaml

   schedule:
     climb:
       Ueas: {value: [230, 252], units: kn}      # interpolated across the phase
       vs:   {value: [2300, 400], units: ft/min}
     cruise:
       Ueas: {value: [252, 252], units: kn}
       vs:   {value: 0, units: ft/min}           # constant

Each continuation step takes ``description``, and optionally ``parameters`` and ``schedule``
overrides. Steps are partial by design -- they override only what they name -- but a step's
schedule entries still require both ``Ueas`` and ``vs``.

``optimization``
----------------

Present makes the case an optimization study; absent makes it a sizing run.

``optimization.driver``
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Key
     - Default
     - Meaning
   * - ``name``
     - ``SLSQP``
     - ``SLSQP`` (SciPy, always available) or a pyOptSparse optimizer such as ``IPOPT`` or
       ``SNOPT``. See :doc:`optimization` for why the shipped case uses IPOPT.
   * - ``maxiter``
     - 50
     - Optimizer iteration limit.
   * - ``tol``
     - 1e-6
     - Optimizer convergence tolerance.
   * - ``derivative_mode``
     - ``fwd``
     - ``auto``, ``fwd`` or ``rev``. Forward is right here: a handful of design variables, many
       vector responses.
   * - ``options``
     - ``{}``
     - Extra settings passed through to pyOptSparse, on top of cdadt's per-optimizer defaults.

``optimization.objective``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   objective: {name: total_fuel, units: kg, sense: minimize, ref: 2.0e4}

``name`` is either a response name any discipline reports -- ``total_fuel``, ``MTOW``,
``takeoff_field_length`` -- or a raw path inside the box. Response names are preferred: they
carry their own units. ``sense`` is ``minimize`` or ``maximize``.

``ref`` scales the objective to order one and is not decoration. SLSQP takes its
finite-difference step and its convergence test on the *scaled* objective, so an objective of
order 1e4 is effectively converged before it starts. IPOPT scales from its own gradient and does
not need it.

``optimization.design_variables``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   design_variables:
     - {name: ac|geom|wing|S_ref, lower: 90.0, upper: 180.0, units: m**2}
     - {name: ac|geom|wing|AR, lower: 7.0, upper: 13.0}

``name`` must be an independent variable of the box -- checked against a cheap probe before
anything expensive runs, so a misspelling is a message with suggestions. ``ref`` defaults to the
larger of the two bounds, which is nearly always the right order of magnitude and is the
difference between an optimizer that converges and one that stalls when a wing area in square
metres shares a design space with an aspect ratio.

Bounds are an engineering statement, not a numerical one: they should be the range over which
the empirical correlations inside the box are defensible for the class of aircraft being
designed, not the range over which the code happens to run.

``optimization.requirements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   requirements:
     - type: balanced_field_length
       limit: 8000.0
       units: ft
       regulation: 14 CFR 25.113
       source: Design field length, 8000 ft dry runway at sea level, ISA

``type``, ``limit``, ``regulation`` and ``source`` are all required. The last two are required
because a constraint in a certification-driven study that cannot name its regulation and where
its number came from is a constraint nobody can defend. Use ``regulation: design`` -- and say so
in ``source`` -- for a programme decision rather than a rule.

Any other key is passed to the requirement class as an option: ``phase`` for a throttle limit,
``response`` and ``sense`` for the generic ``response_limit``. See :doc:`certification` for the
requirement types available and what each one constrains.

Validation
----------

The case file is checked in three passes, each of which names what is wrong:

1. **Structure**, when the file is read: unknown keys, missing keys, non-numeric values,
   unscheduled phases, half-specified schedules, inverted bounds, duplicate design variables,
   unknown senses and unknown derivative modes.
2. **Against the box**, before an optimization runs: every design variable must be an
   independent variable it publishes, and the objective and every constraint must be a quantity
   it produces. Checked on a three-node probe, which costs a fraction of a second.
3. **Against the box**, when the study builds: every ``aircraft`` and mission parameter must be
   settable, and every ``initial_guesses`` name must at least exist.
