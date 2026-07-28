.. _verification:

***************************
Verification and validation
***************************

cdadt is intended to produce numbers that support engineering conclusions. That standard is
only meaningful if it is possible to say precisely *what* has been demonstrated about any
given result. This page defines the vocabulary the test suite uses and the rules it obeys.

Classes of claim
================

Every test carries a pytest marker and restates its class of claim in its docstring. The
marker is not a category for filtering convenience -- it is a statement about the strength
of the evidence the test provides.

``unit``
   Exercises a single cdadt class in isolation. Proves the class does what its own
   docstring says. Proves nothing about the assembled model or about physical correctness.

``contract``
   Verifies the cdadt/OpenConcept interface against a real, built OpenConcept model. These
   are the tests that catch interface drift when the OpenConcept dependency changes. They
   prove the two sides connect; they say nothing about whether the physics is right.

``derivative``
   Verifies analytic partials and totals against complex step. Proves the derivatives an
   optimizer will consume match the functions actually being evaluated.

``integration``
   Builds and runs an assembled multi-discipline model. Proves the pieces converge
   together. A converged model can still be converging to the wrong answer, which is what
   the next class is for.

``validation``
   Compares a cdadt result against an external reference -- the OpenConcept example
   implementation, or a published hand calculation with a citation in the test docstring.
   **This is the only class of test that supports a claim of physical correctness.**

``regression``
   Pins a previously converged result so that drift is caught. Proves the answer has not
   changed; proves nothing about whether it was ever right.

Rules the suite obeys
=====================

**No test doubles for dependencies.** A test that claims to exercise OpenConcept
instantiates and runs the real OpenConcept component. Mocking is permitted only for
filesystem and I/O concerns, never for physics. A test that runs a stand-in and reports a
pass is a test that measures the stand-in.

**No stubs.** A test that cannot yet be satisfied fails loudly. It is never a bare ``pass``
and never a silent ``skip``. Skips that are genuinely unavoidable are reported explicitly
in the run summary.

**Exhaustive, not sampled.** Contract and coverage tests iterate the *full* declared
variable set and assert on every entry. Spot-checking a handful of names produces a gate
that verifies names rather than behavior, and reports success for a set it never examined.

**A reference value used as a default is still a hardcode.** Certification factors and
method constants are required from configuration. A test asserts that omitting one raises,
rather than silently falling back to a value sourced from a textbook. Sourcing a number
does not make it a default.

**No vacuous assertions.** Every test asserts on a value that would change if the code
broke. A test that only checks that an object was constructed, or that a name exists,
provides a green result for a broken model.

**The dependency stays read-only.** ``test_openconcept_integrity.py`` fails the suite if the
OpenConcept working tree has uncommitted modifications, and an AST scan over cdadt's own
source fails the suite if any cdadt module assigns onto an OpenConcept symbol. The rule
that cdadt never modifies OpenConcept is therefore enforced mechanically rather than by
review.

Running the suite
=================

.. code-block:: bash

   pytest -q -m "not slow"                # fast development loop
   pytest -q                              # everything, including reference validation
   pytest -q -m validation -v             # only the physical-correctness claims
   pytest -q --cov=cdadt --cov-report=term-missing

The environment gate
====================

``cdadt/tests/test_environment.py`` runs first in the suite and checks the toolchain by
using it rather than by importing it: it builds a real ``FullMissionWithReserve``, asks
IPOPT to solve a constrained problem with a known analytic optimum, and confirms
OpenConcept resolves to a git working tree so that the editable install is proven to be in
effect. A failure here means results from any later test are not trustworthy, so the
environment is treated as part of the verification chain rather than as setup.
