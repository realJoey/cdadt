"""The tube-and-wing jet transport discipline set.

Seven disciplines, each delegating to an OpenConcept-backed provider. This function is the
one place the modeling choices for this class of aircraft are recorded; swapping a provider
-- an aerostructural drag polar for the empirical buildup, say -- is an edit here and
nowhere else.
"""

from __future__ import annotations

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.discipline import Discipline
from cdadt.disciplines import (
    Aerodynamics,
    EmptyWeight,
    Geometry,
    MassBookkeeping,
    MaximumLift,
    Propulsion,
    Stability,
)
from cdadt.providers.openconcept import (
    FuelBurnMassProvider,
    JetTransportDragProvider,
    JetTransportEmptyWeightProvider,
    JetTransportMaximumLiftProvider,
    RubberizedTurbofanProvider,
    TailVolumeCoefficientProvider,
    TrapezoidalGeometryProvider,
)

__all__ = ["jet_transport_disciplines"]


def jet_transport_disciplines(config: AircraftConfiguration, engine_deck: str) -> list[Discipline]:
    """Return the discipline set for a tube-and-wing jet transport.

    Parameters
    ----------
    config : AircraftConfiguration
        Configuration every provider reads its method constants from. Each provider
        validates its own requirements at construction, so a configuration missing a
        constant fails here with the name of what is absent.
    engine_deck : str
        Engine deck for the propulsion provider: ``"CFM56"`` or ``"N3"``. Required, with no
        default -- which engine an aircraft has is not something to inherit silently.

    Returns
    -------
    list of Discipline
        Four aircraft-scoped disciplines (geometry, stability, maximum lift, empty weight)
        and three phase-scoped ones (aerodynamics, propulsion, mass). The
        :class:`~cdadt.mission.sizing.SizingLoop` separates them by scope; the caller does
        not need to.

    Raises
    ------
    MissingConfigurationError
        If any provider's required configuration is absent.
    ValueError
        If ``engine_deck`` is not a deck OpenConcept provides.
    """
    return [
        # -------------- Aircraft-scoped: properties of the design --------------
        Geometry(TrapezoidalGeometryProvider(config), config),
        Stability(TailVolumeCoefficientProvider(config), config),
        MaximumLift(JetTransportMaximumLiftProvider(config), config),
        EmptyWeight(JetTransportEmptyWeightProvider(config), config),
        # -------------- Phase-scoped: functions of the flight condition --------------
        Aerodynamics(JetTransportDragProvider(config), config),
        Propulsion(RubberizedTurbofanProvider(config, deck=engine_deck), config),
        MassBookkeeping(FuelBurnMassProvider(config), config),
    ]
