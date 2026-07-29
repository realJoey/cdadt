"""The component that evaluates a cdadt loads model at every node of a mission phase.

This is the join between two worlds. On one side is
:class:`~cdadt.models.loads.AerodynamicLoads`: plain Python, numpy, no dependency, testable on
its own. On the other is OpenMDAO, which wants a component with declared inputs, outputs and
partials, instantiated inside a group it controls.

The component owns no physics. It reads the black box's flight conditions and geometry, hands
them to the model as a :class:`~cdadt.models.loads.FlightCondition` and a
:class:`~cdadt.models.planform.TrapezoidalPlanform`, and publishes the drag force the trajectory
consumes. Swapping the model swaps the aerodynamics and changes nothing else.

Notes
-----
The model is constructed once per :meth:`~AerodynamicLoadsComp.compute`, from the input values as
they stand. That is deliberate -- the span efficiency and the zero-lift drag are *variables* of
the black box, not configuration, and a model holding stale copies of them would silently report
the drag of a design the optimizer has already moved away from. It also means **model
construction must be cheap**: it happens at every Newton iteration. A model whose construction is
expensive should cache the expensive part on the class, as
:class:`~cdadt.adapter.avl.OpenAVLLoads` does with its lattice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import openmdao.api as om

from cdadt.models.coefficients import AeroCoefficients
from cdadt.models.loads import AerodynamicLoads, FlightCondition
from cdadt.models.planform import TrapezoidalPlanform

__all__ = ["AerodynamicLoadsComp"]


class AerodynamicLoadsComp(om.ExplicitComponent):
    """Evaluate an aerodynamic loads model over a phase and publish the drag force.

    Options
    -------
    num_nodes : int
        Analysis points in the phase.
    loads_factory : callable
        Called with ``(span_efficiency, zero_lift_drag)`` and returning an
        :class:`~cdadt.models.loads.AerodynamicLoads`. A factory rather than an instance because
        the two arguments are black-box variables that move under the optimizer.
    publish_coefficients : bool
        Also publish ``CD`` and the moment coefficients as outputs. Default ``False``; the
        mission needs only ``drag``, and an unconnected output on every phase is noise in an N2
        diagram. Turned on where the coefficients are wanted for reporting.
    """

    def initialize(self) -> None:
        """Declare the options that make this component specific to a phase and a model."""
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("loads_factory", types=object)
        self.options.declare("publish_coefficients", default=False, types=bool)

    def setup(self) -> None:
        """Declare the flight conditions and geometry read, and the drag published."""
        nodes = self.options["num_nodes"]
        rows = np.arange(nodes)

        self.add_input("fltcond|CL", shape=(nodes,))
        self.add_input("fltcond|q", shape=(nodes,), units="N/m**2")
        self.add_input("fltcond|M", shape=(nodes,))
        self.add_input("fltcond|h", shape=(nodes,), units="m")
        self.add_input("ac|geom|wing|S_ref", shape=(1,), units="m**2")
        self.add_input("ac|geom|wing|AR", shape=(1,))
        self.add_input("ac|geom|wing|c4sweep", shape=(1,), units="deg")
        self.add_input("ac|geom|wing|taper", shape=(1,))
        self.add_input("ac|aero|polar|e", shape=(1,))
        self.add_input("CD0", shape=(nodes,))

        self.add_output("drag", shape=(nodes,), units="N")
        if self.options["publish_coefficients"]:
            self.add_output("CD", shape=(nodes,))

        # Drag at a node depends on that node's flight condition, and on every scalar.
        self.declare_partials("drag", ["fltcond|CL", "fltcond|q", "CD0"], rows=rows, cols=rows)
        self.declare_partials(
            "drag", ["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|aero|polar|e"], rows=rows, cols=np.zeros(nodes)
        )
        if self.options["publish_coefficients"]:
            self.declare_partials("CD", ["fltcond|CL", "CD0"], rows=rows, cols=rows)
            self.declare_partials("CD", ["ac|geom|wing|AR", "ac|aero|polar|e"], rows=rows, cols=np.zeros(nodes))

    # -- evaluation ----------------------------------------------------------------------

    def _model_and_geometry(self, inputs: Any) -> tuple[AerodynamicLoads, TrapezoidalPlanform, FlightCondition]:
        """Build the model, the wing and the flight condition from the current inputs.

        The wing is built first and handed to the model, because a model may need the *shape* and
        not merely its area and aspect ratio -- a vortex lattice is built on the sections.
        """
        planform = TrapezoidalPlanform(
            area=float(inputs["ac|geom|wing|S_ref"][0]),
            aspect_ratio=float(inputs["ac|geom|wing|AR"][0]),
            sweep=float(inputs["ac|geom|wing|c4sweep"][0]),
            taper=float(inputs["ac|geom|wing|taper"][0]),
        )
        model = self.options["loads_factory"].build(
            planform=planform,
            span_efficiency=float(inputs["ac|aero|polar|e"][0]),
            zero_lift_drag=np.asarray(inputs["CD0"], dtype=float),
        )
        condition = FlightCondition(
            CL=np.asarray(inputs["fltcond|CL"], dtype=float),
            mach=np.asarray(inputs["fltcond|M"], dtype=float),
            altitude=np.asarray(inputs["fltcond|h"], dtype=float),
            dynamic_pressure=np.asarray(inputs["fltcond|q"], dtype=float),
        )
        return model, planform, condition

    def compute(self, inputs: Any, outputs: Any) -> None:
        """Evaluate the loads model and publish the drag force."""
        model, planform, condition = self._model_and_geometry(inputs)
        coefficients: AeroCoefficients = model.coefficients(condition, planform)

        outputs["drag"] = coefficients.CD * condition.dynamic_pressure * planform.area
        if self.options["publish_coefficients"]:
            outputs["CD"] = coefficients.CD

    def compute_partials(self, inputs: Any, partials: Any) -> None:
        """Publish analytic derivatives, taken from the model rather than differenced.

        The model supplies the gradients of its own drag *coefficient*; the product rule for
        ``drag = CD q S`` belongs here, because ``q`` and ``S`` are the component's business and
        not the model's.
        """
        model, planform, condition = self._model_and_geometry(inputs)
        gradients = model.drag_gradients(condition, planform)
        coefficients = model.coefficients(condition, planform)

        pressure = condition.dynamic_pressure
        area = planform.area

        partials["drag", "fltcond|CL"] = gradients["CL"] * pressure * area
        partials["drag", "CD0"] = gradients["CD0"] * pressure * area
        partials["drag", "ac|aero|polar|e"] = gradients["e"] * pressure * area
        partials["drag", "ac|geom|wing|AR"] = gradients["AR"] * pressure * area
        partials["drag", "fltcond|q"] = coefficients.CD * area
        partials["drag", "ac|geom|wing|S_ref"] = coefficients.CD * pressure

        if self.options["publish_coefficients"]:
            partials["CD", "fltcond|CL"] = gradients["CL"]
            partials["CD", "CD0"] = gradients["CD0"]
            partials["CD", "ac|aero|polar|e"] = gradients["e"]
            partials["CD", "ac|geom|wing|AR"] = gradients["AR"]
