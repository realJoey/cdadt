"""Concrete physics implementations behind cdadt's disciplines.

A provider is a strategy: it says *how* a discipline computes what it declares. Providers
are grouped by where their physics comes from.

:mod:`cdadt.providers.openconcept`
    Providers that wrap components from the OpenConcept library. These are the only cdadt
    modules outside :mod:`cdadt.mission.blackbox` permitted to import OpenConcept, and they
    wrap it -- they never modify it.

Adding a provider from a different source means adding a package alongside this one. The
disciplines and the mission do not change.
"""
