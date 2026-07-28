"""A minimal but real cdadt aircraft model, for exercising the OpenConcept contract.

The disciplines here are genuine :class:`~cdadt.core.discipline.Discipline` and
:class:`~cdadt.core.provider.Provider` implementations built from real OpenMDAO components,
and they read every constant from a real
:class:`~cdadt.core.configuration.AircraftConfiguration`. They are deliberately
low-fidelity -- a fixed drag polar, a linear thrust lapse, a constant fuel flow -- because
the property under test is the *interface*, not the physics.

They are not test doubles for OpenConcept. Every contract test builds a real
``FullMissionWithReserve`` and lets it instantiate these classes exactly as it would
instantiate the production ones. Physical results from this model are meaningless and no
test asserts on them; the physics is validated separately against OpenConcept's own B738
example.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.discipline import Discipline
from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet
from cdadt.mission.contract import DRAG, THRUST, WEIGHT

SIMPLE_CONFIG_DATA = {
    "ac": {
        "geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}},
        "aero": {
            "CD0": {"value": 0.02},
            "induced_factor": {"value": 0.05},
            "CLmax_TO": {"value": 2.2},
        },
        "propulsion": {
            "sea_level_thrust": {"value": 220e3, "units": "N"},
            "thrust_lapse_per_meter": {"value": 5.0e-5, "units": "m**-1"},
            "specific_fuel_flow": {"value": 1.5e-5, "units": "kg/s/N"},
        },
        "weights": {"MTOW": {"value": 79e3, "units": "kg"}},
    }
}


def simple_config() -> AircraftConfiguration:
    """Return a configuration sufficient for every provider in this module."""
    return AircraftConfiguration(SIMPLE_CONFIG_DATA)


class SimpleDragPolarProvider(Provider):
    """Drag from a fixed parabolic polar, ``CD = CD0 + k * CL**2``."""

    @property
    def name(self) -> str:
        return "simple_drag_polar"

    @property
    def reference(self) -> str:
        return "Fixed parabolic drag polar; interface fixture only, not a validated method."

    def provides(self) -> VariableSet:
        return VariableSet([DRAG])

    def requires(self) -> VariableSet:
        return VariableSet(
            [
                Variable("fltcond|CL", None, vectorized=True),
                Variable("fltcond|q", "N * m**-2", vectorized=True),
                Variable("ac|geom|wing|S_ref", "m**2"),
            ]
        )

    def validate_configuration(self) -> None:
        self.config.require_all(["ac|aero|CD0", "ac|aero|induced_factor"])

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        group.add_subsystem(
            "polar",
            om.ExecComp(
                "drag = (CD0 + k * CL**2) * q * S_ref",
                drag={"val": np.ones(num_nodes), "units": "N"},
                CL={"val": np.full(num_nodes, 0.5)},
                q={"val": np.full(num_nodes, 1.0e4), "units": "N * m**-2"},
                S_ref={"val": 1.0, "units": "m**2"},
                CD0={"val": self.config.scalar("ac|aero|CD0")},
                k={"val": self.config.scalar("ac|aero|induced_factor")},
                has_diag_partials=True,
            ),
            promotes_inputs=[("CL", "fltcond|CL"), ("q", "fltcond|q"), ("S_ref", "ac|geom|wing|S_ref")],
            promotes_outputs=["drag"],
        )


class SimpleThrustProvider(Provider):
    """Thrust with a linear altitude lapse, scaled by throttle and by active propulsors."""

    @property
    def name(self) -> str:
        return "simple_thrust"

    @property
    def reference(self) -> str:
        return "Linear thrust lapse with altitude; interface fixture only, not a validated method."

    def provides(self) -> VariableSet:
        return VariableSet([THRUST, Variable("fuel_flow", "kg/s", vectorized=True)])

    def requires(self) -> VariableSet:
        return VariableSet(
            [
                Variable("throttle", None, vectorized=True),
                Variable("fltcond|h", "m", vectorized=True),
                Variable("propulsor_active", None, vectorized=True),
            ]
        )

    def validate_configuration(self) -> None:
        self.config.require_all(
            [
                "ac|propulsion|sea_level_thrust",
                "ac|propulsion|thrust_lapse_per_meter",
                "ac|propulsion|specific_fuel_flow",
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        # propulsor_active is 0.0 for the failed-engine phases and 1.0 otherwise. Two
        # engines, so one failed leaves half the thrust available.
        group.add_subsystem(
            "thrust_calc",
            om.ExecComp(
                "thrust = T_sl * (1.0 - lapse * h) * throttle * (0.5 + 0.5 * propulsor_active)",
                thrust={"val": np.ones(num_nodes), "units": "N"},
                throttle={"val": np.ones(num_nodes)},
                propulsor_active={"val": np.ones(num_nodes)},
                h={"val": np.zeros(num_nodes), "units": "m"},
                T_sl={"val": self.config.scalar("ac|propulsion|sea_level_thrust", units="N"), "units": "N"},
                lapse={
                    "val": self.config.scalar("ac|propulsion|thrust_lapse_per_meter", units="m**-1"),
                    "units": "m**-1",
                },
                has_diag_partials=True,
            ),
            promotes_inputs=["throttle", "propulsor_active", ("h", "fltcond|h")],
            promotes_outputs=["thrust"],
        )
        group.add_subsystem(
            "fuel_flow_calc",
            om.ExecComp(
                "fuel_flow = sfc * thrust",
                fuel_flow={"val": np.ones(num_nodes), "units": "kg/s"},
                thrust={"val": np.ones(num_nodes), "units": "N"},
                sfc={
                    "val": self.config.scalar("ac|propulsion|specific_fuel_flow", units="kg/s/N"),
                    "units": "kg/s/N",
                },
                has_diag_partials=True,
            ),
            promotes_inputs=["thrust"],
            promotes_outputs=["fuel_flow"],
        )


class SimpleFuelWeightProvider(Provider):
    """Mass as takeoff mass less integrated fuel burn.

    Uses OpenConcept's own ``Integrator``, which is what makes fuel burn accumulate across
    phases: OpenConcept's ``PhaseGroup`` finds ``Integrator`` instances inside the aircraft
    model and links their duration and endpoint states automatically.
    """

    @property
    def name(self) -> str:
        return "simple_fuel_weight"

    @property
    def reference(self) -> str:
        return "Takeoff mass less integrated fuel burn; interface fixture only."

    def provides(self) -> VariableSet:
        return VariableSet([WEIGHT])

    def requires(self) -> VariableSet:
        return VariableSet(
            [
                Variable("fuel_flow", "kg/s", vectorized=True),
                Variable("ac|weights|MTOW", "kg"),
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        from openconcept.utilities import AddSubtractComp, Integrator

        integrator = group.add_subsystem(
            "fuel_burn_integ",
            Integrator(num_nodes=num_nodes, diff_units="s", method="simpson", time_setup="duration"),
            promotes_inputs=["fuel_flow"],
        )
        integrator.add_integrand("fuel_burn", rate_name="fuel_flow", rate_units="kg/s", lower=0.0, upper=1e6)

        group.add_subsystem(
            "weight_calc",
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
        group.connect("fuel_burn_integ.fuel_burn", "weight_calc.fuel_burn")


class _Named(Discipline):
    """A discipline whose only specialization is its name."""

    def __init__(self, provider, config, name):
        self._name = name
        super().__init__(provider, config)

    @property
    def name(self) -> str:
        return self._name


def simple_disciplines(config: AircraftConfiguration | None = None) -> list[Discipline]:
    """Return a discipline set that satisfies the OpenConcept aircraft-model contract.

    Parameters
    ----------
    config : AircraftConfiguration, optional
        Configuration to build the providers from. Defaults to :func:`simple_config`.

    Returns
    -------
    list of Discipline
        Aerodynamics, propulsion, and weights, producing drag, thrust, and mass.
    """
    config = config or simple_config()
    return [
        _Named(SimpleDragPolarProvider(config), config, "aerodynamics"),
        _Named(SimpleThrustProvider(config), config, "propulsion"),
        _Named(SimpleFuelWeightProvider(config), config, "weights"),
    ]
