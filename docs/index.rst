.. _cdadt_index:

*****
cdadt
*****

**cdadt** is a Certification Driven Aircraft Design Tool: an aircraft sizing and
optimization framework in which the certification basis is a first-class modeling object
rather than a set of constraints bolted on after the fact.

Guiding principles
==================

**The certification basis drives the design.** Regulatory requirements
(14 CFR Part 25 §25.113 balanced field length, §25.121 one-engine-inoperative climb
gradients, §25.119 and §25.125 landing performance, and thrust/throttle margins) are
declared as ``Requirement`` objects. Each one knows its
regulation citation, the model variables it reads, the components it contributes, and the
constraint it registers with the optimizer. A run emits a traceability matrix linking every
regulation to the constraint that enforced it and the margin achieved.

**Every discipline is an object.** Aerodynamics, propulsion, weights, geometry, and
stability are classes that encapsulate their own state and declare what they provide and
require. There is no global state and no free-function discipline math. Each discipline
delegates its physics to a swappable *provider*, so a low-fidelity empirical buildup and a
high-fidelity aerostructural analysis satisfy the same interface.

**The mission analysis is a black box.** Full mission sizing -- balanced-field takeoff,
climb, cruise, descent, Part 25 reserves, and loiter -- is supplied by
`OpenConcept <https://github.com/mdolab/openconcept>`_. cdadt consumes it through a single
boundary class and **never modifies OpenConcept**. What that boundary is, and why it is a
contract rather than a wall, is documented in :ref:`blackbox`.

**Nothing is claimed that is not verified.** Every test in cdadt states which class of
claim it makes. Results are validated against the OpenConcept reference implementation and
against published hand calculations; see :ref:`verification`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Framework

   architecture
   blackbox
   certification
   optimization

.. toctree::
   :maxdepth: 2
   :caption: Verification and validation

   verification
   validation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
