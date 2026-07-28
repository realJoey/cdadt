"""Weight providers: the jet-transport empty-weight buildup, and mission mass bookkeeping."""

from __future__ import annotations

import openmdao.api as om
from openconcept.utilities import AddSubtractComp, Integrator
from openconcept.weights import JetTransportEmptyWeight

from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet
from cdadt.mission.contract import WEIGHT

__all__ = ["FuelBurnMassProvider", "JetTransportEmptyWeightProvider"]

#: Geometry, propulsion and weight inputs OpenConcept's jet-transport buildup reads.
_EMPTY_WEIGHT_INPUTS = [
    Variable("ac|num_passengers_max", None, description="Maximum passenger count"),
    Variable("ac|num_flight_deck_crew", None, description="Flight deck crew count"),
    Variable("ac|num_cabin_crew", None, description="Cabin crew count"),
    Variable("ac|cabin_pressure", "psi", description="Cabin pressure"),
    Variable("ac|aero|Mach_max", None, description="Maximum operating Mach number"),
    Variable("ac|aero|Vstall_land", "kn", description="Landing configuration stall speed"),
    Variable("ac|geom|wing|S_ref", "m**2", description="Wing reference area"),
    Variable("ac|geom|wing|AR", None, description="Wing aspect ratio"),
    Variable("ac|geom|wing|c4sweep", "rad", description="Wing quarter-chord sweep"),
    Variable("ac|geom|wing|taper", None, description="Wing taper ratio"),
    Variable("ac|geom|wing|toverc", None, description="Wing thickness-to-chord"),
    Variable("ac|geom|hstab|S_ref", "m**2", description="Horizontal stabilizer area"),
    Variable("ac|geom|hstab|AR", None, description="Horizontal stabilizer aspect ratio"),
    Variable("ac|geom|hstab|c4sweep", "rad", description="Horizontal stabilizer sweep"),
    Variable("ac|geom|hstab|c4_to_wing_c4", "m", description="Horizontal tail moment arm"),
    Variable("ac|geom|vstab|S_ref", "m**2", description="Vertical stabilizer area"),
    Variable("ac|geom|vstab|AR", None, description="Vertical stabilizer aspect ratio"),
    Variable("ac|geom|vstab|c4sweep", "rad", description="Vertical stabilizer sweep"),
    Variable("ac|geom|vstab|toverc", None, description="Vertical stabilizer thickness-to-chord"),
    Variable("ac|geom|vstab|c4_to_wing_c4", "m", description="Vertical tail moment arm"),
    Variable("ac|geom|fuselage|height", "m", description="Fuselage height"),
    Variable("ac|geom|fuselage|length", "m", description="Fuselage length"),
    Variable("ac|geom|fuselage|S_wet", "m**2", description="Fuselage wetted area"),
    Variable("ac|geom|maingear|length", "m", description="Main landing gear length"),
    Variable("ac|geom|maingear|num_wheels", None, description="Main gear wheel count"),
    Variable("ac|geom|maingear|num_shock_struts", None, description="Main gear shock strut count"),
    Variable("ac|geom|nosegear|length", "m", description="Nose landing gear length"),
    Variable("ac|geom|nosegear|num_wheels", None, description="Nose gear wheel count"),
    Variable("ac|propulsion|engine|rating", "lbf", description="Sea-level static thrust per engine"),
    Variable("ac|propulsion|num_engines", None, description="Number of engines"),
    Variable("ac|weights|MTOW", "kg", description="Maximum takeoff mass"),
    Variable("ac|weights|MLW", "kg", description="Maximum landing mass"),
    Variable("ac|weights|W_fuel_max", "kg", description="Maximum fuel mass"),
]


class JetTransportEmptyWeightProvider(Provider):
    """Operating empty weight from a jet-transport component buildup.

    Aircraft-scoped. The buildup sums wing, empennage, fuselage, landing gear, nacelle,
    engine, furnishing and equipment weights from geometry and design weights.

    Maximum landing weight is derived here rather than required as an input, because it is
    otherwise circular: MLW is a fraction of MTOW, and MTOW is what the sizing loop solves
    for. It enters only the landing-gear weight estimate.

    Notes
    -----
    **Required configuration**, with no default:
    ``ac|weights|MLW_fraction_of_MTOW``, maximum landing mass as a fraction of maximum
    takeoff mass. OpenConcept's ``B738_sizing`` example hardcodes 0.8.
    """

    @property
    def name(self) -> str:
        """Return ``"jet_transport_empty_weight"``."""
        return "jet_transport_empty_weight"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "OpenConcept JetTransportEmptyWeight, a component weight buildup following Roskam "
            "and Raymer transport-category correlations."
        )

    def provides(self) -> VariableSet:
        """Return the operating empty weight and the derived maximum landing weight."""
        return VariableSet(
            [
                Variable("ac|weights|OEW", "kg", description="Operating empty mass"),
                Variable("ac|weights|MLW", "kg", description="Maximum landing mass"),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the geometry, propulsion and weight inputs the buildup reads.

        ``ac|weights|MLW`` is excluded: this provider produces it rather than reading it.
        """
        return VariableSet([v for v in _EMPTY_WEIGHT_INPUTS if v.name != "ac|weights|MLW"])

    def validate_configuration(self) -> None:
        """Require the landing-weight fraction; there is no default for it."""
        self.config.require_all(["ac|weights|MLW_fraction_of_MTOW"])

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the landing-weight estimate and the empty-weight buildup to ``group``."""
        group.add_subsystem(
            "landing_weight",
            AddSubtractComp(
                output_name="ac|weights|MLW",
                input_names=["ac|weights|MTOW"],
                units="kg",
                scaling_factors=[self.config.scalar("ac|weights|MLW_fraction_of_MTOW")],
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["ac|weights|MLW"],
        )

        group.add_subsystem(
            "empty_weight",
            JetTransportEmptyWeight(),
            promotes_inputs=[variable.name for variable in _EMPTY_WEIGHT_INPUTS],
            promotes_outputs=[("OEW", "ac|weights|OEW")],
        )


class FuelBurnMassProvider(Provider):
    """Instantaneous mass as takeoff mass less the fuel burned so far.

    Phase-scoped. Uses OpenConcept's :class:`~openconcept.utilities.math.integrals.Integrator`
    deliberately: OpenConcept's ``PhaseGroup`` walks the aircraft model for ``Integrator``
    instances, connects the phase duration to them, and its ``TrajectoryGroup`` links their
    endpoint states across phases. That is the mechanism by which fuel burned in climb
    carries into cruise. Integrating fuel any other way would restart the accumulation in
    every phase, and the total fuel that closes the sizing loop would be the loiter fuel
    alone -- a converged, wrong answer.
    """

    @property
    def name(self) -> str:
        """Return ``"fuel_burn_mass"``."""
        return "fuel_burn_mass"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "Simpson's-rule integration of fuel flow using OpenConcept's Integrator, with "
            "cross-phase state linking supplied by OpenConcept's PhaseGroup and TrajectoryGroup."
        )

    def provides(self) -> VariableSet:
        """Return the instantaneous mass."""
        return VariableSet([WEIGHT])

    def requires(self) -> VariableSet:
        """Return the fuel flow to integrate and the takeoff mass to subtract from."""
        return VariableSet(
            [
                Variable("fuel_flow", "kg/s", vectorized=True, description="Total fuel flow"),
                Variable("ac|weights|MTOW", "kg", description="Maximum takeoff mass"),
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the fuel integrator and the mass subtraction to ``group``.

        The integrator is named ``fuel_burn_integ`` because that name appears in the
        variable paths cdadt reads back from the mission, declared in
        :data:`cdadt.mission.contract.MISSION_OUTPUTS`.
        """
        integrator = group.add_subsystem(
            "fuel_burn_integ",
            Integrator(num_nodes=num_nodes, diff_units="s", method="simpson", time_setup="duration"),
            promotes_inputs=["fuel_flow"],
        )
        integrator.add_integrand(
            "fuel_burn",
            rate_name="fuel_flow",
            rate_units="kg/s",
            lower=0.0,
            upper=1e6,
        )

        group.add_subsystem(
            "mass_calc",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_burn"],
                scaling_factors=[1, -1],
                vec_size=[1, num_nodes],
                units="kg",
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["weight"],
        )
        group.connect("fuel_burn_integ.fuel_burn", "mass_calc.fuel_burn")
