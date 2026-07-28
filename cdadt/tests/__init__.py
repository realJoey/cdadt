"""Test suite for cdadt.

The suite is organized by the *class of claim* each test makes, declared with a pytest
marker and restated in every test docstring:

``unit``
    Exercises a single cdadt class in isolation. Proves the class does what its own
    docstring says; proves nothing about the assembled model.
``contract``
    Verifies the cdadt/OpenConcept interface contract against a real, built OpenConcept
    model. Catches interface drift when OpenConcept changes.
``derivative``
    Verifies analytic partials and totals against complex step.
``integration``
    Builds and runs an assembled multi-discipline model.
``validation``
    Compares a cdadt result against an external reference (the OpenConcept example) or a
    published hand calculation. This is the only class of test that supports a claim of
    physical correctness.
``regression``
    Pins a previously converged result so drift is caught.

Standing rules for everything in this package:

- No test doubles for dependencies. A test that claims to exercise OpenConcept
  instantiates and runs the real OpenConcept component.
- No stubs. A test that cannot yet be satisfied fails; it is never a bare ``pass``.
- Exhaustive, not sampled. Contract tests iterate the full declared variable set.
- No vacuous assertions. Every test asserts on a value that changes if the code breaks.
"""
