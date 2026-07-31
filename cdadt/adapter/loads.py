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
construction must be cheap**: it happens at every Newton iteration.

A model whose construction is *not* cheap keeps the expensive part in the ``workspace`` its caller
injects, as :class:`~cdadt.adapter.avl.OpenAVLLoads` does. This component does not own that
workspace and does not know what is in it -- it is created by the analysis group, whose lifetime is
a study, and passed straight through. That is deliberate: a store owned by this component would be
re-created for each of the fourteen phases, and a store at module scope would be the global the
project's brief forbids.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

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
    loads_factory : type[AerodynamicLoads]
        The model class to install. Its :meth:`~cdadt.models.loads.AerodynamicLoads.build` is
        called with the planform, the span efficiency and the zero-lift drag as they currently
        stand; a class rather than an instance because those are black-box variables that move
        under the optimizer, and because the class states -- through
        :attr:`~cdadt.models.loads.AerodynamicLoads.DRAG_DEPENDS_ON` -- which partials this
        component should declare.
    workspace : object, optional
        Passed to the model's ``build`` unread and unexamined. Somewhere for a model to keep results
        too expensive to recompute per evaluation; see
        :meth:`~cdadt.models.loads.AerodynamicLoads.build`.

    Notes
    -----
    Only ``drag`` is published. The model computes all six coefficients, but the mission consumes
    the force alone, and an output nothing connects to is noise in an N2 diagram of 826 of them.
    The rest are reachable through the model itself when a study wants them.
    """

    #: The black-box variables that hold for the whole phase, mapped to the gradient name the loads
    #: model answers under. Everything else the component reads varies node by node.
    SCALAR_INPUTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "ac|geom|wing|S_ref": "area",
            "ac|geom|wing|AR": "AR",
            "ac|geom|wing|c4sweep": "sweep",
            "ac|geom|wing|taper": "taper",
            "ac|aero|polar|e": "e",
        }
    )

    def initialize(self) -> None:
        """Declare the options that make this component specific to a phase and a model."""
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("loads_factory", types=object)
        self.options.declare("workspace", default=None, types=object, allow_none=True)
        # Filled in setup, once the installed model can be asked what its drag depends on.
        self._scalar_partials: tuple[str, ...] = ()

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

        # Drag at a node depends on that node's flight condition, and on the scalars the installed
        # model says its drag reaches. Asking the model rather than declaring all of them is what
        # keeps the sparsity truthful: for a vortex lattice the taper term is the largest of the
        # four, and for a parabolic polar it does not exist.
        self.declare_partials("drag", ["fltcond|CL", "fltcond|q", "CD0"], rows=rows, cols=rows)
        self._scalar_partials = self._declared_scalars()
        self.declare_partials("drag", list(self._scalar_partials), rows=rows, cols=np.zeros(nodes))

    def _declared_scalars(self) -> tuple[str, ...]:
        """Return the phase-wide inputs this component declares partials against.

        The reference area is always included: it scales the drag coefficient into a force whatever
        the model, so the derivative is non-zero even for a model whose *coefficient* does not see
        it.
        """
        reaches = frozenset(self.options["loads_factory"].DRAG_DEPENDS_ON)
        return tuple(
            name for name, gradient in self.SCALAR_INPUTS.items() if gradient in reaches or name == "ac|geom|wing|S_ref"
        )

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
            workspace=self.options["workspace"],
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

    def compute_partials(self, inputs: Any, partials: Any) -> None:
        """Publish analytic derivatives, taken from the model rather than differenced.

        The model supplies the gradients of its own drag *coefficient*; the product rule for
        ``drag = CD q S`` belongs here, because ``q`` and ``S`` are the component's business and
        not the model's.

        The reference area appears twice and both terms are kept. It scales the coefficient into a
        force, which every model shares, and it may also change the coefficient itself, which is a
        model's own business -- a vortex lattice re-normalises by the new area, a parabolic polar
        does not. Dropping the second term would silently be right for one model and wrong for the
        other.
        """
        model, planform, condition = self._model_and_geometry(inputs)
        gradients = model.drag_gradients(condition, planform)
        coefficients = model.coefficients(condition, planform)

        pressure = condition.dynamic_pressure
        area = planform.area

        partials["drag", "fltcond|CL"] = gradients["CL"] * pressure * area
        partials["drag", "CD0"] = gradients["CD0"] * pressure * area
        partials["drag", "fltcond|q"] = coefficients.CD * area
        for name in self._scalar_partials:
            partials["drag", name] = gradients[self.SCALAR_INPUTS[name]] * pressure * area
        partials["drag", "ac|geom|wing|S_ref"] += coefficients.CD * pressure
