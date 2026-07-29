The case file
=============

A cdadt study is defined by one YAML file, not by a script. The file names the black box, every
design variable, everything written into the box before it is converged, the ladder that
converges it, and -- for an optimization -- the driver, the constraints and the objective.

The layout follows OpenConcept's own B738 run scripts block for block, so that anyone who can
read ``B738.py`` can read a cdadt case:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Section
     - What it is in OpenConcept's own scripts
   * - ``design_variables``
     - ``B738.py`` line 86 -- *"Define a bunch of design variables and airplane-specific
       parameters"* -- and the ``dv_comp.add_output_from_dict(...)`` calls that follow it.
   * - ``initial_conditions``
     - ``set_values(prob, num_nodes)`` in ``B738.py``, plus the first half of
       ``set_mission_profile(prob)`` in ``B738_sizing.py``. Same names.
   * - ``continuation``
     - The second half of ``set_mission_profile``: the ``run_model()`` calls that converge an
       easy mission before the design one is attempted.
   * - ``driver``, ``constraints``, ``objective``
     - The ``add_constraint`` / ``add_objective`` calls in the examples that optimize
       (``B738_aerostructural.py``, ``B738_VLM_drag.py``).

There is deliberately no separate "optimization" section. A variable becomes free for the driver
by gaining an ``optimize:`` entry *where it is already declared*, so a study that frees one more
variable differs from its sizing run by three lines, and no number is ever written twice.

Every section rejects keys it does not recognise. A silently ignored key in a configuration file
means the run succeeds and answers a different question than the one that was asked.

Skeleton
--------

.. code-block:: yaml

   black_box:          # required
   solver:             # optional
   mission_path:       # optional; default "mission"
   design_variables:   # required
   initial_conditions: # optional
   continuation:       # optional
   driver:             # optional
   constraints:        # optional
   objective:          # optional; present makes it an optimization study

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
       never imported by cdadt, never modified, never subclassed. See :doc:`blackbox`.
   * - ``num_nodes``
     - yes
     - Analysis points per mission phase. **Must be odd**: the box integrates fuel burn with
       Simpson's rule, which needs 2N + 1 points. 21 for the shipped sizing case; 11 for the
       optimization, where the grid is traded against iteration count.

``solver``
----------

Configures the Newton solve that closes the weight loop and the mission's own balances together.
All keys optional; the defaults are OpenConcept's own sizing run script's.

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

``mission_path``
----------------

Name of the mission subsystem inside the box. Default ``mission``. It is what lets
``initial_conditions`` write ``cruise|h0`` where the box calls it ``mission.cruise|h0``.

``design_variables``
--------------------

One entry per variable the box publishes, keyed by the name it publishes it under:

.. code-block:: yaml

   design_variables:
     ac|geom|wing|S_ref:
       value: 124.6
       units: m**2
       source: b737.org.uk technical specifications
       optimize: {lower: 90.0, upper: 180.0}     # omit to hold it fixed

.. list-table::
   :header-rows: 1
   :widths: 16 14 70

   * - Key
     - Required
     - Meaning
   * - ``value``
     - yes
     - The number, in ``units``. A list is a vector value, element by element.
   * - ``units``
     - no
     - OpenMDAO unit string (``m**2``, ``lbf``, ``kn``, ``deg``). Omit for a dimensionless
       quantity such as aspect ratio.
   * - ``source``
     - no, but
     - Where the number came from. Technically optional; in practice required, and the shipped
       cases have one for every parameter. A number with no provenance is indistinguishable from
       a guess in the report that quotes it.
   * - ``optimize``
     - no
     - Present frees the variable for the driver. Absent holds it at ``value``.

Every name here must be an **independent variable** of the box. Quantities the box computes --
``ac|weights|OEW``, ``ac|geom|hstab|S_ref``, ``ac|aero|CLmax_TO`` -- are outputs, and setting one
would be overwritten by the next solve. cdadt checks the whole list against the built model and
names every offender at once, with suggestions. ``cdadt inspect <case>`` lists what is available.

The ``optimize`` entry
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Default
     - Meaning
   * - ``lower``, ``upper``
     - --
     - The interval the variable may move in. A scalar, or a list to bound a vector variable
       element by element.
   * - ``indices``
     - all
     - Which elements of a vector variable are free.
   * - ``ref``, ``ref0``, ``scaler``, ``adder``
     - ``ref`` = larger bound
     - OpenMDAO driver scaling, passed through unchanged.

The ``ref`` default is nearly always the right order of magnitude, and is the difference between
an optimizer that converges and one that stalls when a wing area in square metres shares a design
space with an aspect ratio.

Bounds are an engineering statement, not a numerical one: they should be the range over which the
empirical correlations inside the box are defensible for the class of aircraft being designed,
not the range over which the code happens to run.

``initial_conditions``
----------------------

Everything written into the box before it is converged, using the names OpenConcept's own
``set_values`` uses. Mission-level values, the per-phase speed and vertical-speed schedules, and
the solver's starting guesses all live here, because to the box they are the same operation:

.. code-block:: yaml

   initial_conditions:
     mission_range:      {value: 2800, units: nmi}
     cruise|h0:          {value: 35000, units: ft}
     climb.fltcond|Ueas: {value: [230, 252], units: kn}
     climb.fltcond|vs:   {value: [2300, 400], units: ft/min}
     ac|weights|MTOW:    {value: 50.0e3, units: kg}      # a starting guess, not a design

Only ``value`` and ``units`` are accepted. A name is resolved first as written and then under
``mission_path``, so ``cruise|h0`` and ``mission.cruise|h0`` are both legal and a name that
resolves neither way is an error naming both attempts.

A value may be a single number (constant across the phase), two numbers (interpolated across it,
exactly as ``np.linspace`` does in OpenConcept's run script), or exactly as many numbers as the
variable's own shape. Any other length is an error rather than being broadcast, because
broadcasting would quietly fly a different mission.

Writing a value onto an *output* is meaningful here and only here. Maximum takeoff weight is what
the weight closure solves for, so it can never be a design variable; writing a value onto it
before the first solve is where Newton starts, and on a sizing loop that is frequently the
difference between converging and not. These decide whether the solver converges, not what it
converges to.

``continuation``
----------------

A list of progressively harder missions, each converged before the next is attempted. The box is
a Newton-solved implicit system, and started cold on a 2800 nmi mission at FL350 it does not
converge:

.. code-block:: yaml

   continuation:
     - description: short low-altitude mission, gentle descent
       initial_conditions:
         mission_range:      {value: 500, units: nmi}
         cruise|h0:          {value: 5000, units: ft}
         descent.fltcond|vs: {value: -800, units: ft/min}

     - description: design range and altitude, still with the gentle descent
       initial_conditions:
         descent.fltcond|vs: {value: -800, units: ft/min}

A rung takes ``description`` and ``initial_conditions``, nothing else. Rungs are partial by
design: each overrides only what it names, on top of the case's own ``initial_conditions``, which
are re-applied before every rung. The ladder is a route to the answer and not part of it --
:mod:`tests.test_verification_reproducibility` walks three different ladders and requires them to
reach the same aircraft.

``driver``
----------

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

``constraints``
---------------

A list, each entry being one ``add_constraint`` call plus the two fields OpenConcept's examples
have nowhere to put -- the regulation the number comes from, and the source of that number:

.. code-block:: yaml

   constraints:
     - name: takeoff_field_length
       upper: 8000.0
       units: ft
       regulation: 14 CFR 25.113
       source: Design field length, 8000 ft dry runway at sea level, ISA
       title: Balanced field length within the runway available

     - name: climb_throttle          # a plain design bound, no provenance claimed
       lower: 0.01
       upper: 1.05

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Key
     - Required
     - Meaning
   * - ``name``
     - yes
     - A response name any discipline reports -- see :doc:`interface` -- or a raw black-box
       path. Response names are preferred: they carry their own units.
   * - ``lower``, ``upper``, ``equals``
     - one of them
     - Any form ``add_constraint`` accepts: one-sided, two-sided, or equality.
   * - ``units``
     - no
     - Units the bounds are stated in. Defaults to the response's own.
   * - ``indices``
     - no
     - Which elements of a vector response are constrained.
   * - ``linear``
     - no
     - Whether the constraint is linear in the design variables. Default ``false``.
   * - ``ref``, ``ref0``, ``scaler``, ``adder``
     - no
     - Driver scaling. ``ref`` defaults to the magnitude of the bound, without which a climb
       gradient in hundredths of a radian is numerically invisible next to a field length in
       thousands of feet.
   * - ``regulation``, ``source``
     - no
     - Provenance. Supplying **both** is what puts a constraint in the traceability matrix as
       certification evidence; supplying neither reports it as a design constraint. See
       :doc:`certification`.
   * - ``title``
     - no
     - One line naming the constraint in the report. Defaults to ``name``.

A vector response is one constraint, not one per node, and the value reported is the governing
one: the least margin, whichever side of the bound it is near.

``objective``
-------------

Present makes the case an optimization study; absent makes it a sizing run.

.. code-block:: yaml

   objective: {name: total_fuel, units: kg, sense: minimize, ref: 2.0e4}

``name`` is a response name or a raw path, ``sense`` is ``minimize`` or ``maximize``, ``index``
picks one element of a vector output, and ``ref``/``ref0``/``scaler``/``adder`` are OpenMDAO's
scaling arguments.

``ref`` is not decoration. SLSQP takes its finite-difference step and its convergence test on the
*scaled* objective, so an objective of order 1e4 is effectively converged before it starts. IPOPT
scales from its own gradient and does not need it.

Validation
----------

The case file is checked in three passes, each of which names what is wrong:

1. **Structure**, when the file is read: unknown keys, missing keys, non-numeric values, an even
   ``num_nodes``, inverted or missing bounds, an ``equals`` on a design variable, unknown senses
   and unknown derivative modes.
2. **Against the box**, before an optimization runs: every freed variable must be an independent
   variable it publishes, and the objective and every constraint must be a quantity it produces.
   Checked on a cheaply built probe, which costs a fraction of a second rather than failing
   inside ``setup`` after a minute of building.
3. **Against the box**, when the study builds: every design variable and every initial condition
   must resolve to something the box accepts, either as written or under ``mission_path``.
