"""The weights discipline: the operating empty weight buildup."""

from __future__ import annotations

from typing import ClassVar

from openconcept.utilities import AddSubtractComp
from openconcept.weights import JetTransportEmptyWeight

from cdadt.disciplines.base import AircraftDiscipline

__all__ = ["Weights"]


class Weights(AircraftDiscipline):
    """Operating empty weight from an empirical component buildup.

    :class:`~openconcept.weights.JetTransportEmptyWeight` sums wing, tail, fuselage, gear,
    nacelle, engine, systems, furnishing and operational-item weights from transport-category
    correlations. It is the discipline that makes the sizing loop a loop: empty weight depends
    on maximum takeoff weight, and maximum takeoff weight is empty weight plus payload plus
    fuel.

    Maximum landing weight is estimated as a fraction of maximum takeoff weight. It enters
    only the landing-gear weight correlation, so the estimate does not need to be sharp -- but
    it does need to exist, because requiring the user to state it before the aircraft has been
    sized asks for a number that is an output.

    Inputs
    ------
    ac|geom|*, ac|propulsion|*, ac|aero|Mach_max, ac|aero|Vstall_land : float
        Airframe description (scalar).
    ac|num_passengers_max, ac|num_flight_deck_crew, ac|num_cabin_crew, ac|cabin_pressure : float
        Cabin description (scalar).
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, kg). The loop closure supplies this.
    ac|weights|W_fuel_max : float
        Maximum fuel load, which sizes the fuel system (scalar, kg).

    Outputs
    -------
    ac|weights|OEW : float
        Operating empty weight (scalar, kg).
    ac|weights|MLW : float
        Maximum landing weight (scalar, kg).

    Notes
    -----
    :attr:`landing_weight_fraction` is an assumption, stated as a class attribute so it can be
    replaced by subclassing when a real landing-weight requirement is known.
    """

    discipline_name: ClassVar[str] = "weights"

    #: Maximum landing weight as a fraction of maximum takeoff weight. 0.8 is the estimate
    #: OpenConcept's B738 sizing example uses; it feeds the landing-gear weight only.
    landing_weight_fraction: ClassVar[float] = 0.8

    #: Every input the empty-weight buildup reads, promoted so that one design parameter set
    #: at the top of the model reaches it.
    BUILDUP_INPUTS: ClassVar[tuple[str, ...]] = (
        "ac|num_passengers_max",
        "ac|num_flight_deck_crew",
        "ac|num_cabin_crew",
        "ac|cabin_pressure",
        "ac|aero|Mach_max",
        "ac|aero|Vstall_land",
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|taper",
        "ac|geom|wing|toverc",
        "ac|geom|hstab|S_ref",
        "ac|geom|hstab|AR",
        "ac|geom|hstab|c4sweep",
        "ac|geom|hstab|c4_to_wing_c4",
        "ac|geom|vstab|S_ref",
        "ac|geom|vstab|AR",
        "ac|geom|vstab|c4sweep",
        "ac|geom|vstab|toverc",
        "ac|geom|vstab|c4_to_wing_c4",
        "ac|geom|fuselage|height",
        "ac|geom|fuselage|length",
        "ac|geom|fuselage|S_wet",
        "ac|geom|maingear|length",
        "ac|geom|maingear|num_wheels",
        "ac|geom|maingear|num_shock_struts",
        "ac|geom|nosegear|length",
        "ac|geom|nosegear|num_wheels",
        "ac|propulsion|engine|rating",
        "ac|propulsion|num_engines",
        "ac|weights|MTOW",
        "ac|weights|MLW",
        "ac|weights|W_fuel_max",
    )

    def setup(self) -> None:
        """Add the landing-weight estimate and the empty-weight buildup."""
        self.add_subsystem(
            "landing_weight",
            AddSubtractComp(
                output_name="ac|weights|MLW",
                input_names=["ac|weights|MTOW"],
                units="kg",
                scaling_factors=[self.landing_weight_fraction],
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["ac|weights|MLW"],
        )

        self.add_subsystem(
            "empty_weight",
            JetTransportEmptyWeight(),
            promotes_inputs=list(self.BUILDUP_INPUTS),
            promotes_outputs=[("OEW", "ac|weights|OEW")],
        )
