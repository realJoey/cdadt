.. _optimization:

************
Optimization
************

:class:`~cdadt.optimization.problem.DesignProblem` is the one place a cdadt optimization is
assembled. It owns discipline assembly, the sizing loop, the certification basis, the design
variables, the objective and the driver. Nothing else assembles a model: a design study whose
problem definition is spread across a run script is a study whose problem definition nobody
can state.

.. code-block:: python

   design_problem = DesignProblem(
       disciplines=jet_transport_disciplines(config, engine_deck="CFM56"),
       config=config,
       certification=CertificationBasis([...]),
       driver=IpoptDriver(config),
       objective="total_fuel",
   )
   problem = design_problem.build()
   result = design_problem.run(problem)
   print(result.report())

The objective has no default. What a design is being optimized for is the most consequential
choice in a study, and it would be the worst possible place to save a line.

"It finished" is not "it succeeded"
===================================

An OpenMDAO ``run_driver`` call reports that it returned, not that it converged. A run that
hit its iteration limit, or that failed to restore feasibility, still yields a design vector,
a full set of mission results, and a traceability matrix. Those are indistinguishable from an
optimum unless something checks.

That is what :class:`~cdadt.optimization.driver.OptimizerDriver` exists for -- not to make
optimizers interchangeable, which OpenMDAO already does, but to require each one to report
:meth:`~cdadt.optimization.driver.OptimizerDriver.outcome`. IPOPT's exit ``0`` is an optimum;
its exit ``1``, "solved to acceptable level", is **not** treated as one, because acceptable
means IPOPT relaxed its own tolerances after it stopped making progress.

:meth:`~cdadt.optimization.problem.OptimizationResult.report` leads with the status, before
any numbers. A reader who skims the design variables and never sees that the run stopped
short has been misled by the report's layout rather than by its contents.

Scaling is not cosmetic
=======================

This is worth stating plainly because it decided whether the B738 case worked at all.

The problem spans a taper ratio of 0.25, an engine rating of 27,000 lbf, a climb gradient of
0.03 rad, a field length of 8,000 ft and a fuel burn of 18,000 kg. Presented to an optimizer
unscaled, the field-length constraint appears five orders of magnitude more important than
the climb gradient, and the climb gradient appears satisfied to within noise.

Run that way, IPOPT reached the right answer and then **could not certify it**: it ran 32
iterations, entered feasibility restoration, and exited ``-13``, "Invalid Number Detected".
The design it stopped at was the optimum to four significant figures. Nothing in the numbers
said so.

cdadt therefore non-dimensionalizes all three sides of the problem:

* **Constraints** by ``1 / |limit|``, in
  :meth:`~cdadt.certification.basis.Requirement.register`. Every requirement's constraint is
  order one regardless of its units.
* **Design variables** by ``1 / max(|lower|, |upper|)``, unless a scaler is configured.
* **The objective** by a configured ``optimization|objective_reference``. There is no
  default: a fuel burn and an empty weight are both in kilograms and differ by a factor of
  three, a takeoff distance in feet by three more.

With that, the same problem exits ``0``, "Solve Succeeded", at the same design.

The B738 result
===============

Minimizing total mission fuel -- design mission plus Part 25 reserves -- with wing area,
aspect ratio, sweep, taper and engine rating free:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - Baseline
     - Optimum
     - Change
   * - Total fuel (kg)
     - 18,594
     - 17,164
     - **−7.7%**
   * - MTOW (kg)
     - 78,341
     - 75,249
     - −3.9%
   * - OEW (kg)
     - 41,747
     - 40,084
     - −4.0%
   * - Wing area (m²)
     - 124.6
     - 131.5
     - +5.5%
   * - Aspect ratio
     - 9.45
     - 11.0 *(bound)*
     - +16%
   * - Engine rating (lbf)
     - 27,000
     - 22,185
     - −17.8%

Ten of ten requirements met, **two active**: the approach-speed limit for category C, and
the climb throttle limit. Those two shaped the design.

That is the result a certification-driven method is supposed to produce. The optimizer did
not simply find a lighter aeroplane -- it found the lightest one that still lands inside
category C and still climbs on the thrust it has. Aspect ratio sitting on its bound says the
remaining improvement is limited by the range the weight correlation was fitted over, not by
the physics.

Running it
==========

.. code-block:: bash

   python examples/optimize_b738.py
   python examples/optimize_b738.py --optimizer SLSQP --objective MTOW

The script chooses only what cannot be a number: which providers, which engine deck, which
requirements are active, and what to minimize. Everything else -- airframe, mission,
certification limits and their sources, design variable bounds, optimizer settings -- lives
in ``configs/b738.yml``.
