"""The models: what OpenConcept builds inside a phase, and what wraps the whole mission.

Two OpenMDAO groups, and the split between them is the black-box boundary.

:class:`JetTransportPhaseModel` is the *input* to the black box that is hardest to see as an
input. OpenConcept's mission does not accept an aircraft as data; it accepts an aircraft model
*class*, which it instantiates once per phase, sized to that phase's node count. Supplying
that class is the documented way to use OpenConcept, and it is the only thing cdadt hands
across the boundary besides numbers.

:class:`SizingModel` is everything outside the black box: the airframe disciplines that are
evaluated once, the weight closure that makes takeoff weight consistent with the fuel burned,
and the mission itself as a single subsystem.
"""

from __future__ import annotations

from typing import ClassVar

import openmdao.api as om
from openconcept.mission import FullMissionWithReserve
from openconcept.utilities import AddSubtractComp

from cdadt.aircraft import AircraftDefinition
from cdadt.disciplines import (
    Aerodynamics,
    Geometry,
    HighLift,
    MassProperties,
    Propulsion,
    Stability,
    Weights,
)

__all__ = ["JetTransportPhaseModel", "Part25Aerodynamics", "Part25PhaseModel", "SizingModel"]


class JetTransportPhaseModel(om.Group):
    """The aircraft as OpenConcept sees it inside one mission phase.

    OpenConcept constructs this class itself, passing only ``num_nodes`` and ``flight_phase``,
    and requires it to consume the flight condition, the throttle, the lift coefficient and
    the ``ac|`` design parameters, and to produce ``thrust``, ``drag`` and ``weight``. Those
    three outputs are the whole contract; everything else is internal.

    The disciplines are a class attribute rather than a constructor argument because
    OpenConcept controls the construction. Swapping one is a subclass::

        class HighFidelityPhaseModel(JetTransportPhaseModel):
            disciplines = (VLMAerodynamics, Propulsion, MassProperties)

    Options
    -------
    num_nodes : int
        Analysis points in this phase, set by OpenConcept.
    flight_phase : str
        Phase name, set by OpenConcept. Reaches the disciplines, which use it to decide
        whether the aircraft is in takeoff configuration.

    Inputs
    ------
    fltcond|* : float
        Flight condition at each node, from the phase (vector).
    throttle, propulsor_active : float
        Throttle setting and engine-failure flag, from the phase (vector).
    ac|* : float
        Design parameters, promoted down from the top of the model (scalar).

    Outputs
    -------
    thrust, drag, weight : float
        The three quantities every OpenConcept phase requires (vector).
    fuel_burn_final : float
        Fuel burned by the end of this phase (scalar, kg). Not part of OpenConcept's contract;
        read by the sizing loop from the last phase.
    """

    #: Phase-scoped disciplines, in build order: drag, then thrust and fuel flow, then the
    #: fuel-burn integration that turns fuel flow into weight.
    disciplines: ClassVar[tuple[type, ...]] = (Aerodynamics, Propulsion, MassProperties)

    def initialize(self) -> None:
        """Declare the options OpenConcept passes when it builds a phase."""
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points to run")
        self.options.declare(
            "flight_phase", default=None, types=str, allow_none=True, desc="Phase of mission this group lives in"
        )

    def setup(self) -> None:
        """Build each discipline, promoting everything so they couple by name."""
        for discipline in self.disciplines:
            self.add_subsystem(
                discipline.discipline_name,
                discipline(num_nodes=self.options["num_nodes"], flight_phase=self.options["flight_phase"]),
                promotes=["*"],
            )


class Part25Aerodynamics(Aerodynamics):
    """Aerodynamics that also deploys takeoff flaps for the engine-out climb condition.

    14 CFR 25.121(b) specifies the second-segment climb gradient "with the landing gear
    retracted and the takeoff flaps set". OpenConcept's own B738 example evaluates its
    engine-out climb-angle condition with clean drag, which is optimistic for that regulation.

    This subclass adds ``EngineOutClimbAngle`` to the set of phases flown in takeoff
    configuration. It changes the answer -- the gradient falls -- so it is opt-in rather than
    the default, and the default reproduces the reference example exactly.
    """

    takeoff_configuration_phases: ClassVar[frozenset[str]] = Aerodynamics.takeoff_configuration_phases | {
        "EngineOutClimbAngle"
    }


class Part25PhaseModel(JetTransportPhaseModel):
    """Phase model that evaluates the engine-out climb gradient in takeoff configuration.

    Identical to :class:`JetTransportPhaseModel` except for the aerodynamics discipline. Use
    it when 14 CFR 25.121(b) is an active requirement; use the base class when reproducing
    OpenConcept's published results.
    """

    disciplines: ClassVar[tuple[type, ...]] = (Part25Aerodynamics, Propulsion, MassProperties)


class SizingModel(om.Group):
    """The coupled sizing model: airframe disciplines, weight closure, and the mission.

    The circularity this model closes is the whole of aircraft sizing:

    .. math::

       \\mathrm{MTOW} = \\mathrm{OEW}(\\mathrm{MTOW}, \\text{geometry})
                        + W_\\mathrm{payload}
                        + W_\\mathrm{fuel}(\\mathrm{MTOW}, \\text{mission})

    A heavier aircraft burns more fuel, and more fuel makes it heavier. The Newton solver
    attached by :class:`~cdadt.sizing.SizingAnalysis` drives that residual to zero at the same
    time as the mission's own implicit states -- phase durations, throttle settings and the
    decision speed V\\ :sub:`1`.

    Which fuel closes the loop matters. It is the fuel burned by the end of *loiter*, the last
    phase fuel accumulates through, not the block fuel at the end of descent. Closing on block
    fuel would size an aircraft that carries no reserves, and would converge just as readily.

    Parameters
    ----------
    aircraft : AircraftDefinition
        Design parameters. Every parameter it contains becomes an independent variable, and
        therefore a candidate optimizer design variable. It must not contain a quantity this
        model computes -- tail areas, MAC, wetted areas, OEW, MLW, MTOW -- because that would
        declare the same variable twice.
    phase_model : type, optional
        Class OpenConcept instantiates inside every phase. Default
        :class:`JetTransportPhaseModel`.
    num_nodes : int, optional
        Analysis points per phase. Must be odd, because the fuel-burn integration uses
        Simpson's rule. Default 11.
    initial_MTOW : float, optional
        Starting guess for maximum takeoff weight, in kg. Default 50e3, as in OpenConcept's
        B738 sizing example. A starting guess, not a tuning parameter: it decides whether the
        solver converges, not what it converges to.
    initial_fuel : float, optional
        Starting guess for total mission fuel, in kg. Default 30e3, likewise.

    Raises
    ------
    ValueError
        If ``num_nodes`` is even.
    """

    #: Aircraft-scoped disciplines, built once above the mission, in build order.
    disciplines: ClassVar[tuple[type, ...]] = (Geometry, Stability, HighLift, Weights)

    #: Subsystem name the OpenConcept mission is added under.
    MISSION = "mission"

    #: Promoted path, relative to the mission, of the fuel burned by the end of the mission.
    #: Loiter is the last phase fuel accumulates through.
    TOTAL_FUEL = "loiter.fuel_burn_final"

    def initialize(self) -> None:
        """Declare the model's options."""
        self.options.declare("aircraft", types=AircraftDefinition, desc="Design parameters")
        self.options.declare("phase_model", default=JetTransportPhaseModel, desc="Aircraft model class per phase")
        self.options.declare("num_nodes", default=11, types=int, desc="Analysis points per mission phase")
        self.options.declare("initial_MTOW", default=50e3, types=float, desc="Starting guess for MTOW, kg")
        self.options.declare("initial_fuel", default=30e3, types=float, desc="Starting guess for mission fuel, kg")

    def setup(self) -> None:
        """Build the design parameters, the airframe disciplines, the closure and the mission."""
        num_nodes = self.options["num_nodes"]
        if num_nodes % 2 == 0:
            raise ValueError(
                f"num_nodes must be odd because fuel burn is integrated with Simpson's rule; got {num_nodes}."
            )

        aircraft = self.options["aircraft"]
        self.add_subsystem("design_parameters", aircraft.component(), promotes_outputs=["*"])

        for discipline in self.disciplines:
            self.add_subsystem(discipline.discipline_name, discipline(), promotes=["*"])

        self.add_subsystem(
            "takeoff_weight",
            AddSubtractComp(
                output_name="ac|weights|MTOW",
                input_names=["ac|weights|OEW", "ac|weights|W_payload", "ac|weights|W_fuel"],
                units="kg",
                lower=1e-6,
            ),
            promotes_inputs=["ac|weights|OEW", "ac|weights|W_payload"],
            promotes_outputs=["ac|weights|MTOW"],
        )

        self.add_subsystem(
            self.MISSION,
            FullMissionWithReserve(num_nodes=num_nodes, aircraft_model=self.options["phase_model"]),
            promotes_inputs=["ac|*"],
        )

        # Total mission fuel closes the weight loop and sizes the fuel system.
        self.connect(
            f"{self.MISSION}.{self.TOTAL_FUEL}",
            ["takeoff_weight.ac|weights|W_fuel", "ac|weights|W_fuel_max"],
        )

        # Starting points for the two states that drive the coupled solve. MTOW is an output
        # of the closure, so it never receives a value through the input-default path; seeding
        # its own input side is what keeps the first mission evaluation on a real aircraft.
        self.set_input_defaults("ac|weights|MTOW", self.options["initial_MTOW"], units="kg")
        self.set_input_defaults("takeoff_weight.ac|weights|W_fuel", self.options["initial_fuel"], units="kg")
        self.set_input_defaults("ac|weights|W_fuel_max", self.options["initial_fuel"], units="kg")
