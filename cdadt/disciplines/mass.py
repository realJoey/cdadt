"""The mass discipline: fuel burned so far, and the weight the aircraft is flying at."""

from __future__ import annotations

from typing import ClassVar

from openconcept.utilities import AddSubtractComp, Integrator

from cdadt.disciplines.base import PhaseDiscipline

__all__ = ["MassProperties"]


class MassProperties(PhaseDiscipline):
    """Fuel burn and instantaneous weight through one mission phase.

    Integrates the fuel flow the propulsion discipline produces, then subtracts it from the
    maximum takeoff weight:

    .. math::

       W_\\mathrm{fuel}(t) = \\int_0^t \\dot{m}_\\mathrm{fuel}\\,\\mathrm{d}t,
       \\qquad W(t) = \\mathrm{MTOW} - W_\\mathrm{fuel}(t)

    The integration is Simpson's rule over the phase's nodes, which is why the node count
    must be odd. The phase duration is not an input this class connects: OpenConcept's
    :class:`~openconcept.mission.mission_groups.PhaseGroup` finds every integrator anywhere in
    a phase and wires the phase duration to it during ``configure``. The duration is itself a
    solver state -- climb runs until the cruise altitude is reached, cruise until the range is
    flown -- so fuel burn and mission length are solved together.

    Phases are chained by OpenConcept's ``link_phases``, which connects each phase's final
    integrated state to the next phase's initial value. Fuel therefore accumulates across the
    whole mission, and the total burn is the value at the end of the last phase.

    Inputs
    ------
    fuel_flow : float
        Total fuel flow at each node (vector, kg/s).
    ac|weights|MTOW : float
        Maximum takeoff weight (scalar, kg).

    Outputs
    -------
    fuel_burn : float
        Fuel burned since the start of the mission, at each node (vector, kg).
    fuel_burn_final : float
        Fuel burned by the end of this phase (scalar, kg).
    weight : float
        Aircraft weight at each node (vector, kg). Required by every OpenConcept phase.
    """

    discipline_name: ClassVar[str] = "mass"

    def setup(self) -> None:
        """Add the fuel-burn integrator and the weight bookkeeping."""
        num_nodes = self.options["num_nodes"]

        integrator = self.add_subsystem(
            "fuel_burn_integ",
            Integrator(num_nodes=num_nodes, diff_units="s", method="simpson", time_setup="duration"),
            promotes_inputs=["fuel_flow"],
            promotes_outputs=["fuel_burn", "fuel_burn_final"],
        )
        integrator.add_integrand("fuel_burn", rate_name="fuel_flow", rate_units="kg/s", lower=0.0, upper=1e6)

        self.add_subsystem(
            "weight",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_burn"],
                scaling_factors=[1, -1],
                vec_size=[1, num_nodes],
                units="kg",
            ),
            promotes_inputs=["ac|weights|MTOW", "fuel_burn"],
            promotes_outputs=["weight"],
        )
