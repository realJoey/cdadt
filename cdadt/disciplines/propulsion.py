"""The propulsion discipline: a rubberized turbofan deck scaled to the installed engines."""

from __future__ import annotations

from typing import ClassVar

from openconcept.propulsion import RubberizedTurbofan
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp

from cdadt.disciplines.base import PhaseDiscipline

__all__ = ["Propulsion"]


class Propulsion(PhaseDiscipline):
    """Thrust and fuel flow at every node of one mission phase.

    :class:`~openconcept.propulsion.RubberizedTurbofan` is a surrogate of pyCycle data for one
    engine, scaled continuously by its sea-level static rating -- which is what makes the
    rating usable as a design variable rather than a catalogue choice. This discipline
    evaluates that deck once and multiplies its thrust and fuel flow by the number of engines
    still running:

    .. math::

       n_\\mathrm{active} = n_\\mathrm{engines} - 1 + \\texttt{propulsor\\_active}

    ``propulsor_active`` is 1 in normal flight and 0 where OpenConcept models the critical
    engine failed -- the abort and engine-out climb conditions -- so the same expression
    covers both without a branch.

    Inputs
    ------
    throttle, fltcond|h, fltcond|M, propulsor_active : float
        Throttle setting, altitude, Mach number and the engine-failure flag at each node,
        supplied by the OpenConcept phase (vector).
    ac|propulsion|engine|rating : float
        Sea-level static thrust rating of one engine (scalar, lbf).
    ac|propulsion|num_engines : float
        Number of engines installed (scalar).

    Outputs
    -------
    thrust : float
        Total thrust from the running engines at each node (vector, lbf).
    fuel_flow : float
        Total fuel flow of the running engines at each node (vector, kg/s). Read by
        :class:`~cdadt.disciplines.mass.MassProperties`, which integrates it.

    Options
    -------
    engine : str
        Engine deck, ``"CFM56"`` or ``"N3"``. Set on the class rather than per instance,
        because OpenConcept constructs the phase model itself and passes only ``num_nodes``
        and ``flight_phase``; subclass to change it.
    """

    discipline_name: ClassVar[str] = "propulsion"

    #: OpenConcept engine deck this discipline evaluates.
    engine: ClassVar[str] = "CFM56"

    def setup(self) -> None:
        """Add the engine deck, the active-engine count, and the multipliers."""
        num_nodes = self.options["num_nodes"]

        self.add_subsystem(
            "engine_deck",
            RubberizedTurbofan(num_nodes=num_nodes, engine=self.engine),
            promotes_inputs=["throttle", "fltcond|h", "fltcond|M", "ac|propulsion|engine|rating"],
        )

        self.add_subsystem(
            "active_engines",
            AddSubtractComp(
                output_name="num_active_engines",
                input_names=["num_engines", "propulsor_active", "one"],
                vec_size=[1, num_nodes, 1],
                scaling_factors=[1, 1, -1],
            ),
            promotes_inputs=[("num_engines", "ac|propulsion|num_engines"), "propulsor_active"],
        )
        self.set_input_defaults("active_engines.one", 1.0)

        totals = self.add_subsystem(
            "installed_totals",
            ElementMultiplyDivideComp(),
            promotes_outputs=["thrust", "fuel_flow"],
        )
        totals.add_equation(
            output_name="thrust",
            input_names=["thrust_per_engine", "num_active_engines_thrust"],
            vec_size=num_nodes,
            input_units=["lbf", None],
        )
        totals.add_equation(
            output_name="fuel_flow",
            input_names=["fuel_flow_per_engine", "num_active_engines_fuel"],
            vec_size=num_nodes,
            input_units=["kg/s", None],
        )
        self.connect("engine_deck.thrust", "installed_totals.thrust_per_engine")
        self.connect("engine_deck.fuel_flow", "installed_totals.fuel_flow_per_engine")

        # ElementMultiplyDivideComp gives each equation its own inputs, so the one engine
        # count feeds both under two names.
        self.connect(
            "active_engines.num_active_engines",
            ["installed_totals.num_active_engines_thrust", "installed_totals.num_active_engines_fuel"],
        )
