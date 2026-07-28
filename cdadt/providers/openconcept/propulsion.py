"""Propulsion provider wrapping OpenConcept's rubberized turbofan engine deck."""

from __future__ import annotations

import openmdao.api as om
from openconcept.propulsion import RubberizedTurbofan
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp

from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet
from cdadt.mission.contract import THRUST

__all__ = ["RubberizedTurbofanProvider"]


class RubberizedTurbofanProvider(Provider):
    """Installed thrust and fuel flow from a scalable turbofan deck.

    Wraps OpenConcept's :class:`~openconcept.propulsion.RubberizedTurbofan`, a surrogate of
    pyCycle data with a rated-thrust multiplier so engine size can be a continuous design
    variable. The deck gives per-engine thrust and fuel flow; this provider multiplies by
    the number of *operating* engines.

    That multiplication is where ``propulsor_active`` enters. OpenConcept sets it to zero in
    the phases that establish the balanced field length and the §25.121 climb gradient, so
    the number of operating engines is ``num_engines - 1 + propulsor_active``. A provider
    that ignored it would size the aircraft against an all-engines-operating takeoff and
    still converge.

    Parameters
    ----------
    config : AircraftConfiguration
        Configuration for the aircraft being modeled.
    deck : str
        Which engine deck to use: ``"CFM56"`` or ``"N3"``. **Required, with no default.**
        OpenConcept defaults to ``"N3"``; which engine an aircraft has is not something to
        inherit silently. This is a constructor argument rather than a configuration entry
        because it selects a model, not a number --
        :class:`~cdadt.core.configuration.AircraftConfiguration` holds quantities with
        units, and a deck name has neither.

    Raises
    ------
    ValueError
        If ``deck`` is not a deck OpenConcept provides.

    Notes
    -----
    OpenConcept documents both decks as valid only from Mach 0.2 to 0.8 and up to
    35,000 ft. Outside that envelope the surrogate is extrapolating.
    """

    #: Valid engine decks, as declared by OpenConcept's RubberizedTurbofan.
    VALID_DECKS = ("CFM56", "N3")

    def __init__(self, config, deck: str) -> None:
        if deck not in self.VALID_DECKS:
            raise ValueError(
                f"deck is '{deck}', which OpenConcept's RubberizedTurbofan does not provide. "
                f"Valid decks: {list(self.VALID_DECKS)}."
            )
        self._deck = deck
        super().__init__(config)

    @property
    def deck(self) -> str:
        """Return the engine deck this provider uses."""
        return self._deck

    @property
    def name(self) -> str:
        """Return ``"rubberized_turbofan"``."""
        return "rubberized_turbofan"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            f"OpenConcept RubberizedTurbofan, engine deck '{self._deck}': a surrogate of pyCycle data "
            f"with a rated-thrust multiplier. Valid Mach 0.2-0.8 and up to 35,000 ft."
        )

    def provides(self) -> VariableSet:
        """Return total installed thrust and total fuel flow."""
        return VariableSet(
            [
                THRUST,
                Variable("fuel_flow", "kg/s", vectorized=True, description="Total fuel flow, all operating engines"),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the flight conditions, controls and engine sizing this deck reads."""
        return VariableSet(
            [
                Variable("throttle", None, vectorized=True, description="Throttle setting"),
                Variable("fltcond|h", "m", vectorized=True, description="Altitude"),
                Variable("fltcond|M", None, vectorized=True, description="Mach number"),
                Variable(
                    "propulsor_active",
                    None,
                    vectorized=True,
                    description="1.0 when all propulsors operate, 0.0 when one is failed",
                ),
                Variable("ac|propulsion|engine|rating", "lbf", description="Sea-level static thrust per engine"),
                Variable("ac|propulsion|num_engines", None, description="Number of engines"),
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the engine deck and the operating-engine multipliers to ``group``."""
        group.add_subsystem(
            "engine_deck",
            RubberizedTurbofan(num_nodes=num_nodes, engine=self._deck),
            promotes_inputs=["throttle", "fltcond|h", "fltcond|M", "ac|propulsion|engine|rating"],
        )

        # num_active_engines = num_engines - 1 + propulsor_active, so a failed propulsor
        # removes exactly one engine.
        group.add_subsystem(
            "operating_engines",
            AddSubtractComp(
                output_name="num_active_engines",
                input_names=["num_engines", "propulsor_active", "one"],
                vec_size=[1, num_nodes, 1],
                scaling_factors=[1, 1, -1],
            ),
            promotes_inputs=[("num_engines", "ac|propulsion|num_engines"), "propulsor_active"],
        )
        group.set_input_defaults("operating_engines.one", 1.0)

        totals = group.add_subsystem(
            "installed_totals", ElementMultiplyDivideComp(), promotes_outputs=["thrust", "fuel_flow"]
        )
        totals.add_equation(
            output_name="thrust",
            input_names=["thrust_per_engine", "num_active_engines_for_thrust"],
            vec_size=num_nodes,
            input_units=["lbf", None],
        )
        totals.add_equation(
            output_name="fuel_flow",
            input_names=["fuel_flow_per_engine", "num_active_engines_for_fuel"],
            vec_size=num_nodes,
            input_units=["kg/s", None],
        )
        group.connect("engine_deck.thrust", "installed_totals.thrust_per_engine")
        group.connect("engine_deck.fuel_flow", "installed_totals.fuel_flow_per_engine")
        # ElementMultiplyDivideComp gives each equation its own inputs, so the same engine
        # count is connected to both rather than promoted to a single shared input.
        group.connect(
            "operating_engines.num_active_engines",
            [
                "installed_totals.num_active_engines_for_thrust",
                "installed_totals.num_active_engines_for_fuel",
            ],
        )
