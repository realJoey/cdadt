"""The only part of cdadt that imports a dependency.

Everywhere else, cdadt names what it drives and loads it at run time, so the boundary cannot
erode. That worked while cdadt computed nothing. Installing cdadt's *own* aerodynamics into
OpenConcept's mission is different: an aircraft model must be an OpenMDAO group that OpenConcept
instantiates, and building one means composing OpenConcept's propulsion and weight blocks around
cdadt's loads.

So exactly one place is allowed to import OpenConcept and openavl, and it is this package. That
is what the project brief asks for -- *"The wrapper should be the only part of CDADT that
directly imports OpenConcept"* -- and a contract test enforces it by allow-listing this package
and no other.

What lives here
---------------

:mod:`cdadt.adapter.loads`
    The OpenMDAO component that evaluates a :class:`~cdadt.models.loads.AerodynamicLoads` over a
    phase and publishes ``drag``.
:mod:`cdadt.adapter.aircraft`
    The per-phase aircraft model OpenConcept's mission instantiates: cdadt's loads, OpenConcept's
    engine deck, fuel integrator and weight bookkeeping.
:mod:`cdadt.adapter.analysis`
    The sizing analysis group that installs the aircraft model into
    ``FullMissionWithReserve`` -- the thing a case file names as its black box.

Nothing here computes physics. The physics is in :mod:`cdadt.models`, which imports neither
dependency and can be tested without them; this package is wiring.
"""

from cdadt.adapter.loads import AerodynamicLoadsComp

__all__ = ["AerodynamicLoadsComp"]
