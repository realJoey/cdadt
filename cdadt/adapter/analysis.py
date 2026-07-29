"""The sizing analysis a case file names, with cdadt's aerodynamics installed in it.

This is the group that makes the substitution possible. ``B738_sizing.py`` builds its mission
with ``aircraft_model=B738AircraftModel`` hardcoded, so no amount of configuration from outside
can change the aerodynamics. Owning the analysis group -- and therefore the line that installs
the aircraft model -- is the whole of what it takes.

Everything except the aerodynamics is OpenConcept's, composed as its own example composes it:
the trapezoidal wing geometry, the tail volume-coefficient sizing, the wetted areas, the maximum
lift estimates, the jet-transport empty-weight buildup, the weight closure, and
``FullMissionWithReserve`` -- the balanced-field takeoff, climb, cruise, descent, the Part 25
reserve diversion and the loiter.

What differs from OpenConcept's own group
-----------------------------------------

Two things, both deliberate.

**The aeroplane comes from the case file.** OpenConcept's group reads its aircraft from a Python
data dictionary shipped inside the library. cdadt declares the same variables as plain
independent variables and lets the case file supply every value, which is what makes a study a
file rather than a source edit -- and what keeps this group from being a copy of a data table
cdadt does not own.

**The aerodynamics is a parameter.** The loads model is named in the case file and resolved to a
class, so a study can fly the same aeroplane through a parabolic polar or a vortex lattice by
changing one line.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import openmdao.api as om
from openconcept.aerodynamics import CleanCLmax, FlapCLmax, ParasiteDragCoefficient_JetTransport
from openconcept.geometry import CylinderSurfaceArea, WingMACTrapezoidal
from openconcept.mission import FullMissionWithReserve
from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing
from openconcept.utilities import AddSubtractComp
from openconcept.weights import JetTransportEmptyWeight

from cdadt.adapter.aircraft import CdadtAircraftModel
from cdadt.loader import ClassSpec
from cdadt.models.loads import AerodynamicLoads, LoadsError

__all__ = ["SizingMissionAnalysis", "jet_transport_zero_lift_drag"]


def jet_transport_zero_lift_drag(num_nodes: int, in_takeoff: bool) -> om.Group:
    """Return OpenConcept's parasite drag buildup, configured for the phase.

    The zero-lift drag is left with OpenConcept on purpose. It is a component-by-component
    empirical buildup -- skin friction, form factors, flap drag, wetted areas -- and
    reimplementing it in cdadt would be copying six hundred lines of somebody else's
    correlations, which the project's rules forbid and which would prove nothing.

    What cdadt owns is what the case file can change: how that zero-lift drag combines with the
    lift-dependent drag its own model computes.
    """
    return ParasiteDragCoefficient_JetTransport(num_nodes=num_nodes, configuration="takeoff" if in_takeoff else "clean")


class SizingMissionAnalysis(om.Group):
    """A full-mission sizing analysis whose aerodynamics are supplied by cdadt.

    Options
    -------
    num_nodes : int
        Analysis points per mission phase. Must be odd; the fuel burn is integrated with
        Simpson's rule.
    aerodynamic_loads : str
        ``"module:ClassName"`` naming the :class:`~cdadt.models.loads.AerodynamicLoads` to fly
        with. Defaults to the parabolic polar, which reproduces what OpenConcept's own group
        computes.

    Notes
    -----
    The loads model is resolved from a string for the same reason the black box itself is: a
    study should be able to change the aerodynamics without editing Python. See
    :class:`~cdadt.loader.ClassSpec`.
    """

    #: The aircraft variables this group accepts, with the units they are declared in. The case
    #: file supplies every value; these are the names and the dimensions, not the aeroplane.
    AIRCRAFT_VARIABLES: tuple[tuple[str, str | None], ...] = (
        ("ac|aero|polar|e", None),
        ("ac|aero|Mach_max", None),
        ("ac|aero|Vstall_land", "m/s"),
        ("ac|aero|airfoil_Cl_max", None),
        ("ac|aero|takeoff_flap_deg", "deg"),
        ("ac|propulsion|engine|rating", "lbf"),
        ("ac|propulsion|num_engines", None),
        ("ac|geom|wing|S_ref", "m**2"),
        ("ac|geom|wing|AR", None),
        ("ac|geom|wing|c4sweep", "deg"),
        ("ac|geom|wing|taper", None),
        ("ac|geom|wing|toverc", None),
        ("ac|geom|hstab|AR", None),
        ("ac|geom|hstab|c4sweep", "deg"),
        ("ac|geom|hstab|taper", None),
        ("ac|geom|hstab|toverc", None),
        ("ac|geom|vstab|AR", None),
        ("ac|geom|vstab|c4sweep", "deg"),
        ("ac|geom|vstab|taper", None),
        ("ac|geom|vstab|toverc", None),
        ("ac|geom|fuselage|length", "m"),
        ("ac|geom|fuselage|height", "m"),
        ("ac|geom|nacelle|length", "m"),
        ("ac|geom|nacelle|diameter", "m"),
        ("ac|geom|maingear|length", "m"),
        ("ac|geom|maingear|num_wheels", None),
        ("ac|geom|maingear|num_shock_struts", None),
        ("ac|geom|nosegear|length", "m"),
        ("ac|geom|nosegear|num_wheels", None),
        ("ac|weights|W_payload", "kg"),
        ("ac|num_passengers_max", None),
        ("ac|num_flight_deck_crew", None),
        ("ac|num_cabin_crew", None),
        ("ac|cabin_pressure", "psi"),
    )

    def initialize(self) -> None:
        """Declare the grid and the aerodynamics."""
        self.options.declare("num_nodes", default=11, types=int)
        self.options.declare("aerodynamic_loads", default="cdadt.models.polar:PolarLoads", types=str)

    def setup(self) -> None:
        """Compose the aeroplane, the weights and the mission."""
        nodes = self.options["num_nodes"]

        self._add_aircraft_variables()
        self._add_geometry()
        self._add_maximum_lift()
        self._add_weights()
        self._add_mission(nodes)
        self._add_solver_seeds()

    # -- the aeroplane, from the case file -----------------------------------------------

    def _add_aircraft_variables(self) -> None:
        """Declare every ``ac|`` variable as an independent variable the case file writes.

        Values are placeholders. cdadt writes the real ones into the built problem, exactly as it
        does for OpenConcept's own group, so nothing about a particular aeroplane is stated here.
        """
        variables = self.add_subsystem("ac_vars", om.IndepVarComp(), promotes_outputs=["*"])
        for name, units in self.AIRCRAFT_VARIABLES:
            variables.add_output(name, val=1.0, units=units)

    def _add_geometry(self) -> None:
        """Wing mean chord, tail areas from volume coefficients, and the wetted areas."""
        self.add_subsystem(
            "tail_lever_arm",
            AddSubtractComp(
                output_name="c4_to_wing_c4", input_names=["fuselage_length"], units="m", scaling_factors=[0.5]
            ),
            promotes_inputs=[("fuselage_length", "ac|geom|fuselage|length")],
        )
        self.connect("tail_lever_arm.c4_to_wing_c4", ["ac|geom|hstab|c4_to_wing_c4", "ac|geom|vstab|c4_to_wing_c4"])

        self.add_subsystem(
            "wing_mean_chord",
            WingMACTrapezoidal(),
            promotes_inputs=[
                ("S_ref", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
            ],
            promotes_outputs=[("MAC", "ac|geom|wing|MAC")],
        )
        self.add_subsystem(
            "vstab_sizing",
            VStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|geom|vstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|vstab|S_ref"],
        )
        self.add_subsystem(
            "hstab_sizing",
            HStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|MAC", "ac|geom|hstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|hstab|S_ref"],
        )

        for name, length, diameter, wetted in (
            ("nacelle_area", "ac|geom|nacelle|length", "ac|geom|nacelle|diameter", "ac|geom|nacelle|S_wet"),
            ("fuselage_area", "ac|geom|fuselage|length", "ac|geom|fuselage|height", "ac|geom|fuselage|S_wet"),
        ):
            self.add_subsystem(
                name,
                CylinderSurfaceArea(),
                promotes_inputs=[("L", length), ("D", diameter)],
                promotes_outputs=[("A", wetted)],
            )

    def _add_maximum_lift(self) -> None:
        """Clean and takeoff maximum lift coefficients, which size the field length."""
        self.add_subsystem(
            "clean_CLmax",
            CleanCLmax(),
            promotes_inputs=["ac|aero|airfoil_Cl_max", "ac|geom|wing|c4sweep"],
            promotes_outputs=[("CL_max_clean", "ac|aero|CLmax_cruise")],
        )
        self.add_subsystem(
            "takeoff_CLmax",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|takeoff_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_TO")],
        )

    def _add_weights(self) -> None:
        """The empty weight buildup and the takeoff weight it closes against."""
        self.add_subsystem(
            "landing_weight",
            AddSubtractComp(
                output_name="ac|weights|MLW", input_names=["ac|weights|MTOW"], units="kg", scaling_factors=[0.8]
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["ac|weights|MLW"],
        )
        self.add_subsystem(
            "empty_weight",
            JetTransportEmptyWeight(),
            promotes_inputs=[name for name, _ in self._empty_weight_inputs()],
            promotes_outputs=[("OEW", "ac|weights|OEW")],
        )
        self.add_subsystem(
            "takeoff_weight",
            AddSubtractComp(output_name="MTOW", input_names=["OEW", "W_payload", "W_fuel"], units="kg", lower=1e-6),
            promotes_inputs=[("OEW", "ac|weights|OEW"), ("W_payload", "ac|weights|W_payload")],
            promotes_outputs=[("MTOW", "ac|weights|MTOW")],
        )

    @staticmethod
    def _empty_weight_inputs() -> tuple[tuple[str, None], ...]:
        """Return the variables the empty-weight buildup reads, as promoted names."""
        names = [
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
        ]
        return tuple((name, None) for name in names)

    # -- the mission, with cdadt's aerodynamics installed --------------------------------

    def _loads_factory(self) -> Callable[..., AerodynamicLoads]:
        """Return the callable that builds the aerodynamic loads model for every phase.

        Raises
        ------
        LoadsError
            If the case file names something that is not an
            :class:`~cdadt.models.loads.AerodynamicLoads`. Caught here, before setup, rather than
            as an attribute error at the first Newton iteration.
        """
        spec = ClassSpec(self.options["aerodynamic_loads"], describes="aerodynamic loads model")
        model_class = spec.resolve(LoadsError)
        if not issubclass(model_class, AerodynamicLoads):
            raise LoadsError(
                f"'{spec.spec}' names {model_class.__name__}, which is not an AerodynamicLoads. "
                f"An aerodynamics model must subclass it so that every study reports which "
                f"model produced its numbers."
            )
        return model_class

    def _add_mission(self, nodes: int) -> None:
        """Install the aircraft model into OpenConcept's full mission with reserves."""
        aircraft_model = partial(
            CdadtAircraftModel,
            loads_factory=self._loads_factory(),
            zero_lift_drag_builder=jet_transport_zero_lift_drag,
        )
        self.add_subsystem(
            "mission",
            FullMissionWithReserve(num_nodes=nodes, aircraft_model=aircraft_model),
            promotes_inputs=["ac|*"],
        )
        self.connect(
            "mission.loiter.fuel_burn_integ.fuel_burn_final",
            ["takeoff_weight.W_fuel", "empty_weight.ac|weights|W_fuel_max"],
        )

    def _add_solver_seeds(self) -> None:
        """Seed the states the weight closure solves for.

        The takeoff weight and the fuel burned appear on both sides of a loop, so the Newton
        solver needs somewhere to start. These are starting points, not answers: the converged
        values are what the study reports.
        """
        self.set_input_defaults("ac|weights|MTOW", 50e3, units="kg")
        for fuel in ("takeoff_weight.W_fuel", "empty_weight.ac|weights|W_fuel_max"):
            self.set_input_defaults(fuel, 30e3, units="kg")

    def __repr__(self) -> str:
        """Return a representation naming the grid and the aerodynamics."""
        return (
            f"SizingMissionAnalysis(num_nodes={self.options['num_nodes']}, "
            f"aerodynamic_loads={self.options['aerodynamic_loads']!r})"
        )
