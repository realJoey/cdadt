"""
cdadt -- A Certification Driven Aircraft Design Tool.

cdadt sizes and optimizes aircraft against an explicit certification basis. Discipline
physics is organized into classes that own their own state (:mod:`cdadt.core`,
:mod:`cdadt.disciplines`), the mission analysis is consumed as a black box
(:mod:`cdadt.mission`), and the regulatory requirements that drive the design are declared
as first-class objects (:mod:`cdadt.certification`).

The mission black box is backed by OpenConcept. OpenConcept is treated as an external,
read-only dependency: cdadt imports it and never modifies, vendors, or monkey-patches it.
See ``docs/blackbox.rst`` for the interface contract.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
