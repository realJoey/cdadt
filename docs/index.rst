cdadt -- a certification-driven aircraft design tool
====================================================

cdadt sizes and optimizes an aircraft against an explicit certification basis.

Two ideas hold the whole thing together.

**The mission analysis is a black box.** Full mission sizing -- balanced-field takeoff, climb,
cruise, descent, 14 CFR Part 25 reserves and loiter -- comes from
`OpenConcept <https://github.com/mdolab/openconcept>`_. cdadt sets its inputs, converges it,
and reads its outputs. It does not modify OpenConcept, subclass it to change behaviour, or
reimplement any part of it. See :doc:`blackbox`.

**Regulations are objects, not comments.** A requirement knows its citation, its limit and
where that limit came from, the quantity it constrains, and any analysis it needs to evaluate
itself. Every run emits a traceability matrix saying which regulations were met, which were
binding, and by how much. See :doc:`certification`.

Everything else follows from wanting those two things to be true.

Validated result
----------------

Run on the shipped Boeing 737-800 case, cdadt reproduces OpenConcept's own published
``B738_sizing.py`` results to solver tolerance -- every quantity, at 21 nodes per phase:

=============================  ==============  ==============
Quantity                       OpenConcept     cdadt
=============================  ==============  ==============
MTOW (kg)                      78345.6435      78345.6435
OEW (kg)                       41748.3258      41748.3258
Block fuel (kg)                15977.0628      15977.0628
Total fuel with reserves (kg)  18597.3177      18597.3177
Balanced field length (ft)     5247.7948       5247.7948
Horizontal tail area (m^2)     27.9332         27.9332
Vertical tail area (m^2)       20.2101         20.2101
=============================  ==============  ==============

The comparison is a test, and the reference is *run* rather than quoted. See
:doc:`validation`, which also states what has **not** been validated.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install
   tutorials
   architecture
   blackbox
   mission
   certification
   optimization
   validation
   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
