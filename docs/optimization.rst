Optimization
============

:class:`~cdadt.optimization.DesignOptimizer` owns the coupling between what may change and
what must hold. It is handed a :class:`~cdadt.sizing.SizingAnalysis`, a set of
:class:`~cdadt.optimization.DesignVariable`\ s, an objective, and a
:class:`~cdadt.certification.CertificationBasis`.

.. code-block:: python

   optimizer = DesignOptimizer(
       analysis=analysis,
       design_variables=[
           DesignVariable("ac|geom|wing|S_ref", lower=90.0, upper=180.0, units="m**2"),
           DesignVariable("ac|geom|wing|AR", lower=7.0, upper=13.0),
           DesignVariable("ac|propulsion|engine|rating", lower=18000.0, upper=34000.0, units="lbf"),
       ],
       objective="total_fuel",
       certification=basis,
       optimizer="IPOPT",
   )
   optimizer.run()
   print(optimizer.report())

Design variables must be independent
------------------------------------

A design variable has to be something the aircraft definition publishes. Anything the model
computes -- tail areas, MAC, OEW, MTOW -- is not free to choose, and OpenMDAO will say so.
This is a feature: the tails are *sized* from the wing, so growing the wing grows the
empennage that stabilizes it and the empty weight follows. Fixed tail areas would let the
optimizer buy wing area for free.

Order of operations
-------------------

1. Build the model. Design variables, constraints and the objective are registered in a hook
   that runs before ``setup``, because OpenMDAO requires all three by then.
2. Converge the baseline with the mission's continuation schedule. This is not optional: the
   optimizer's first function evaluation must start from a converged aircraft, and every later
   one starts from its predecessor.
3. Run the driver.

Because of step 2, the report can compare baseline against optimum without a second run.

Scaling decides the result
--------------------------

Three scalings matter, and two of them are automatic.

**Design variables** are scaled by ``ref``, defaulting to the variable's upper bound. Wing area
in square metres and aspect ratio share a design space; without scaling, the optimizer sees a
1 m² step and a 1-unit step in aspect ratio as the same size.

**Constraints** are scaled by ``1 / |limit|`` in
:meth:`~cdadt.certification.Requirement.register`. See :doc:`certification`.

**The objective** is scaled by ``objective_ref``, and this one is manual. SLSQP takes its
finite-difference step and its convergence test on the scaled objective, so a fuel mass of
order 10\ :sup:`4` kg is effectively converged before it starts -- pass ``objective_ref=2e4``.
IPOPT scales the objective from its own gradient (``nlp_scaling_method: gradient-based``) and
ignores the argument.

Optimizers
----------

``"SLSQP"`` uses SciPy and is always available. Anything else is resolved through pyOptSparse;
``"IPOPT"`` and ``"SNOPT"`` have per-optimizer settings applied. IPOPT is the default in the
shipped example: the sizing model's totals are cheap relative to the Newton solve, so a
limited-memory Hessian approximation with an adaptive barrier is the right trade.

Worked result
-------------

``examples/optimize_b738.py``, minimizing total mission fuel over wing area, aspect ratio,
quarter-chord sweep, taper and engine rating, against the five-requirement basis in
:doc:`certification`, at 11 nodes per phase:

============================  ============  ============  ==========
Quantity                      Baseline      Optimum       Change
============================  ============  ============  ==========
Total fuel (kg)               18596.83      15991.39      **-14.0%**
MTOW (kg)                     78345.02      71959.34      -8.2%
OEW (kg)                      41748.19      37967.95      -9.1%
Wing area (m²)                124.60        99.91         -19.8%
Aspect ratio                  9.45          13.00         +37.6%
Engine rating (lbf)           27000         21911         -18.8%
Balanced field length (ft)    5247.7        6263.7        +19.4%
============================  ============  ============  ==========

Five of five requirements met, one active: the climb throttle limit. That single constraint is
what stopped the engine shrinking further, and it is the requirement that shaped this design.

Two things in that table deserve to be read sceptically rather than celebrated:

* **Aspect ratio went to its upper bound.** The empty-weight correlation penalizes span, but
  not enough to produce an interior optimum within these bounds. A result sitting on a bound
  is a statement about the bound, not about the aerodynamics.
* **Field length grew by 19%** and remains well inside the 8000 ft limit. On a shorter runway
  that constraint would become active and the answer would be a different aircraft. The
  optimum is a function of the certification basis, which is the point of the tool.

Reference
---------

.. automodule:: cdadt.optimization
   :members:
   :noindex:
