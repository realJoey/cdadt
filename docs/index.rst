cdadt
=====

A certification-driven aircraft design tool.

cdadt sizes and optimizes an aircraft against an explicit certification basis. The sizing
itself -- balanced-field takeoff, climb, cruise, descent, 14 CFR Part 25 reserves and loiter --
is performed by an `OpenConcept <https://github.com/mdolab/openconcept>`_ analysis used as a
**black box**: cdadt sets its inputs, converges it, and reads its outputs.

.. code-block:: bash

   cdadt size cases/b738.yaml
   cdadt optimize cases/b738_optimization.yaml
   cdadt inspect cases/b738.yaml

What it is
----------

Three claims, each of which is enforced by the test suite rather than asserted here:

**OpenConcept is a black box.** No cdadt module imports OpenConcept. The sizing analysis is
named in the case file as ``module:ClassName`` and loaded at run time, so there is no
OpenConcept class cdadt can subclass, no component it can re-wire, and no physics it can
quietly reimplement. See :doc:`blackbox`.

**Every discipline is a class.** Geometry, aerodynamics, propulsion, stability, structures,
weights and performance are classes with encapsulated state, and each owns exactly one slice of
the black box's interface: which variables its domain sets, and which responses its domain
reports. Ownership is total and disjoint, checked against the live model. See
:doc:`architecture`.

**A study is a file.** Every design variable, everything written into the box before it is
converged, the continuation ladder, the driver, the objective and the certification constraints
are all declared in one YAML case file, laid out block for block like OpenConcept's own run
scripts. Two studies that ask different questions of the same aeroplane differ only in data. See
:doc:`configuration`.

What it produces
----------------

Sizing the shipped B738 case reproduces OpenConcept's own published example to better than
1e-6 relative, with the reference *run* in the test rather than quoted:

=================================  ===============
Maximum takeoff weight             78,345.6 kg
Operating empty weight             41,748.3 kg
Block fuel                         15,977.1 kg
Fuel with reserves                 18,597.3 kg
Balanced field length              5,247.8 ft
=================================  ===============

Optimizing it against 14 CFR 25.113, 25.121(b) and the engine deck's throttle band, over the
wing planform and the engine rating, cuts fuel with reserves by 11.8% and maximum takeoff weight
by 7.7%. The climb throttle band is the active constraint; neither certification constraint
binds. See :doc:`optimization` and, for what is *not* established, :doc:`validation`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: How it works

   architecture
   blackbox
   configuration
   mission
   aerodynamics
   certification
   optimization
   artifacts

.. toctree::
   :maxdepth: 2
   :caption: Verification and validation

   verification
   validation
   truth

.. toctree::
   :maxdepth: 2
   :caption: Reference

   openconcept
   interface
   developing
   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
