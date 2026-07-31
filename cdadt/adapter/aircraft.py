"""The per-phase aircraft model OpenConcept's mission instantiates.

OpenConcept's mission profiles take the aircraft model as an option -- ``aircraft_model``,
documented as *"OpenConcept-compliant airplane model"* -- and instantiate it once per phase with
``promotes_inputs=["*"], promotes_outputs=["*"]``. The contract is therefore simple and total:
given the phase's flight conditions, throttle and the aircraft's ``ac|`` variables, publish
``drag``, ``thrust``, ``fuel_flow`` and ``weight``.

This class is that contract with **cdadt's aerodynamics in it**. Everything else is
OpenConcept's, used as published and unmodified: the rubberized CFM56 deck, the engine-count
bookkeeping that handles an engine failure, Simpson-rule fuel integration, and the weight that
falls out of it. Only the drag comes from :mod:`cdadt.models`.

Why this class exists at all
----------------------------

``B738_sizing.py`` hardcodes ``aircraft_model=B738AircraftModel`` at the point it builds the
mission, so the aerodynamics cannot be substituted from outside it. Owning the aircraft model is
the smallest change that makes the substitution possible, and it needs no modification to
OpenConcept whatsoever -- ``openconcept/examples/B738_VLM_drag.py`` does the same thing to swap in
a vortex-lattice drag model.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
from openconcept.propulsion import RubberizedTurbofan
from openconcept.utilities import AddSubtractComp, ElementMultiplyDivideComp, Integrator

from cdadt.adapter.loads import AerodynamicLoadsComp
from cdadt.adapter.sections import WingSectionsComp

__all__ = ["CdadtAircraftModel"]


class CdadtAircraftModel(om.Group):
    """One flight phase of an aircraft whose aerodynamics belong to cdadt.

    Options
    -------
    num_nodes : int
        Analysis points in this phase.
    flight_phase : str
        Which phase this is. Only the takeoff phases are distinguished, because the zero-lift
        drag buildup is configured for takeoff flaps there.
    loads_factory : callable
        Builds the :class:`~cdadt.models.loads.AerodynamicLoads` this aircraft flies with. Passed
        down from the analysis group, which reads it from the case file.
    zero_lift_drag_builder : callable or None
        Called with ``(num_nodes, in_takeoff)`` to build the component supplying ``CD0``. ``None``
        means the aircraft has no separate parasite buildup and the loads model produces the
        whole drag coefficient itself.
    wave_drag : bool
        Add OpenConcept's own transonic wave drag on top of the parasite buildup. Default
        ``False``, and the default is load-bearing: OpenConcept's ``B738AircraftModel`` has no wave
        drag, so an aircraft that added it unasked could not reproduce the reference example, and
        the parity anchor that makes every other comparison meaningful would be gone. Requires
        OpenAeroStruct, which is where OpenConcept keeps the component.
    """

    #: Phases that run along the ground, where the flaps are down.
    TAKEOFF_PHASES: tuple[str, ...] = ("v0v1", "v1v0", "v1vr", "rotate")

    def initialize(self) -> None:
        """Declare what makes this model specific to a phase and a study."""
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("flight_phase", default=None, types=(str, type(None)))
        self.options.declare("loads_factory", types=object)
        self.options.declare("zero_lift_drag_builder", default=None, types=(object, type(None)))
        self.options.declare("wave_drag", default=False, types=bool)
        self.options.declare("workspace", default=None, types=object, allow_none=True)

    def setup(self) -> None:
        """Compose the aerodynamics, the propulsion and the weight bookkeeping."""
        nodes = self.options["num_nodes"]
        in_takeoff = self.options["flight_phase"] in self.TAKEOFF_PHASES

        self._add_aerodynamics(nodes, in_takeoff)
        self._add_propulsion(nodes)
        self._add_weight(nodes)

    # -- aerodynamics: cdadt's -----------------------------------------------------------

    def _add_aerodynamics(self, nodes: int, in_takeoff: bool) -> None:
        """Add the zero-lift drag source and cdadt's loads model on top of it."""
        builder = self.options["zero_lift_drag_builder"]
        if builder is None:
            self.add_subsystem("zero_lift_drag", om.IndepVarComp("CD0", val=np.zeros(nodes)))
        else:
            self.add_subsystem("zero_lift_drag", builder(nodes, in_takeoff), promotes_inputs=["*"])

        lift_independent_drag = "zero_lift_drag.CD0"
        if self.options["wave_drag"]:
            lift_independent_drag = self._add_wave_drag(nodes)

        self.add_subsystem(
            "aero_loads",
            AerodynamicLoadsComp(
                num_nodes=nodes,
                loads_factory=self.options["loads_factory"],
                workspace=self.options["workspace"],
            ),
            # The whole planform is promoted, not just area and aspect ratio. A parabolic polar
            # ignores sweep and taper, but a lattice is *built* from them -- and an unpromoted
            # input silently keeps its default, so a missing name here would have the vortex
            # lattice solving an unswept untapered wing while the case file described a swept one.
            promotes_inputs=[
                "fltcond|CL",
                "fltcond|q",
                "fltcond|M",
                "fltcond|h",
                "ac|geom|wing|S_ref",
                "ac|geom|wing|AR",
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|taper",
                "ac|aero|polar|e",
            ],
            promotes_outputs=["drag"],
        )
        self.connect(lift_independent_drag, "aero_loads.CD0")

        # The loads component and OpenConcept's engine deck both read the altitude and the Mach
        # number, and they declare them with different units and defaults. OpenMDAO refuses that
        # ambiguity rather than picking one, which is right -- so the aircraft model states the
        # shared declaration once, here, as OpenMDAO's own error message prescribes. Inside a
        # mission the phase connects a real source and these defaults are never used; standalone,
        # they are what makes the group buildable at all.
        self.set_input_defaults("fltcond|h", val=np.zeros(nodes), units="m")
        self.set_input_defaults("fltcond|M", val=np.zeros(nodes))

    def _add_wave_drag(self, nodes: int) -> str:
        """Add OpenConcept's own transonic wave drag, and return the path of the summed ``CD0``.

        The gap this closes was real and it was ours: neither the vortex lattice nor OpenConcept's
        jet-transport parasite buildup carries a Mach term, so the shipped 737-800 cruised at M 0.785
        -- above the drag-rise Mach of a real wing of that sweep -- with no transonic drag rise
        anywhere in the model. It was documented as an absence.

        It did not have to be. OpenConcept ships ``WaveDragFromSections``, a Korn-equation model
        *"based on the Korn equation"* using *"the same wave drag approximation as OpenAeroStruct"*,
        verified against OpenAeroStruct by OpenConcept's own ``test_wave_drag.py``. So the fix is to
        install the dependency's component rather than to write one: cdadt supplies the sections from
        its planform and adds the result to the parasite drag.

        On the shipped wing this contributes nothing below about M 0.7, 5.0e-5 at the M 0.785 cruise
        -- some 0.2% of the total -- and 8.8e-4 at the M 0.82 maximum, where it is 3%. Small at
        cruise and not small at the edge of the envelope, which is the shape a drag-rise model should
        have and is the reason its absence mattered most where a study would push hardest.
        """
        from openconcept.aerodynamics.openaerostruct import WaveDragFromSections

        self.add_subsystem("wing_sections", WingSectionsComp(), promotes_inputs=["*"])
        self.add_subsystem(
            "wave_drag",
            # specify_area_norm, so the coefficient is normalised by the same reference area the
            # rest of the drag is. Without it the wave drag would be normalised by the region's own
            # planform area and would not be addable to CD0 at all.
            WaveDragFromSections(num_nodes=nodes, num_sections=2, specify_area_norm=True),
            # The component names the wing quantities plainly; the mission names them with the ``ac|``
            # prefix. Renaming on promotion is how OpenConcept's own groups bridge that, and it is
            # what keeps this from being a change to the component.
            promotes_inputs=[
                "fltcond|M",
                "fltcond|CL",
                ("c4sweep", "ac|geom|wing|c4sweep"),
                ("S_ref", "ac|geom|wing|S_ref"),
            ],
        )
        for section_quantity in ("y_sec", "chord_sec", "toverc_sec"):
            self.connect(f"wing_sections.{section_quantity}", f"wave_drag.{section_quantity}")

        self.add_subsystem(
            "total_zero_lift_drag",
            AddSubtractComp(
                output_name="CD0", input_names=["CD0_parasite", "CD_wave"], vec_size=nodes, scaling_factors=[1, 1]
            ),
        )
        self.connect("zero_lift_drag.CD0", "total_zero_lift_drag.CD0_parasite")
        self.connect("wave_drag.CD_wave", "total_zero_lift_drag.CD_wave")
        return "total_zero_lift_drag.CD0"

    # -- propulsion and weight: OpenConcept's, unmodified --------------------------------

    def _add_propulsion(self, nodes: int) -> None:
        """Add the engine deck and the engine-count bookkeeping an engine failure needs."""
        self.add_subsystem(
            "engine_deck",
            RubberizedTurbofan(num_nodes=nodes, engine="CFM56"),
            promotes_inputs=["throttle", "fltcond|h", "fltcond|M", "ac|propulsion|engine|rating"],
        )

        # propulsor_active is 0 for a failed engine and 1 otherwise, so the number still running
        # is (num_engines - 1 + propulsor_active).
        self.add_subsystem(
            "active_engines",
            AddSubtractComp(
                output_name="num_active_engines",
                input_names=["num_engines", "propulsor_active", "one"],
                vec_size=[1, nodes, 1],
                scaling_factors=[1, 1, -1],
            ),
            promotes_inputs=[("num_engines", "ac|propulsion|num_engines"), "propulsor_active"],
        )
        self.set_input_defaults("active_engines.one", 1.0)

        totals = self.add_subsystem("engine_totals", ElementMultiplyDivideComp(), promotes_outputs=["thrust"])
        for output, per_engine, units in (
            ("thrust", "thrust_per_engine", "lbf"),
            ("fuel_flow", "fuel_flow_per_engine", "kg/s"),
        ):
            totals.add_equation(
                output_name=output,
                input_names=[per_engine, f"num_active_{output}"],
                vec_size=nodes,
                input_units=[units, None],
            )
        self.connect("engine_deck.thrust", "engine_totals.thrust_per_engine")
        self.connect("engine_deck.fuel_flow", "engine_totals.fuel_flow_per_engine")
        self.connect(
            "active_engines.num_active_engines",
            ["engine_totals.num_active_thrust", "engine_totals.num_active_fuel_flow"],
        )

    def _add_weight(self, nodes: int) -> None:
        """Integrate fuel burn and subtract it from the takeoff weight."""
        integrator = self.add_subsystem(
            "fuel_burn_integ", Integrator(num_nodes=nodes, diff_units="s", method="simpson", time_setup="duration")
        )
        integrator.add_integrand("fuel_burn", rate_name="fuel_flow", rate_units="kg/s", lower=0.0, upper=1e6)
        self.connect("engine_totals.fuel_flow", "fuel_burn_integ.fuel_flow")

        self.add_subsystem(
            "weight_calc",
            AddSubtractComp(
                output_name="weight",
                input_names=["ac|weights|MTOW", "fuel_burn"],
                scaling_factors=[1, -1],
                vec_size=[1, nodes],
                units="kg",
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["weight"],
        )
        self.connect("fuel_burn_integ.fuel_burn", "weight_calc.fuel_burn")

    def __repr__(self) -> str:
        """Return a representation naming the phase and grid."""
        return f"CdadtAircraftModel({self.options['flight_phase']!r}, num_nodes={self.options['num_nodes']})"
