"""The aerodynamic loads abstraction: the slot a model plugs into.

Shaped after falco's :class:`falco.core.loads.loads.Loads`, which declares one abstract method
turning a flight state into forces and moments. The same idea here, in coefficient form, because
that is what a vortex-lattice method returns and what OpenConcept's mission asks for.

How a model reaches the mission
-------------------------------

A model never talks to OpenConcept. It is handed a :class:`FlightCondition` and a
:class:`Planform` and returns :class:`~cdadt.models.coefficients.AeroCoefficients`;
:mod:`cdadt.adapter` is what installs it into an OpenMDAO group and hands its drag coefficient to
the trajectory. So a new model is a new subclass and nothing else moves -- and a model can be
written, differentiated and tested with neither OpenConcept nor openavl installed.

Two implementations are shipped:

:class:`~cdadt.models.polar.PolarLoads`
    A parabolic drag polar. Deliberately the same equation OpenConcept uses, which is what lets
    it reproduce the reference exactly and prove the surrounding machinery is faithful before any
    new physics rides on it.

:class:`~cdadt.adapter.avl.OpenAVLLoads`
    Vortex-lattice loads from openavl, differentiable with respect to the planform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from cdadt.models.coefficients import AeroCoefficients

__all__ = ["AerodynamicLoads", "FlightCondition", "LoadsError", "Planform"]


class LoadsError(Exception):
    """Raised when an aerodynamic loads model cannot produce what it was asked for."""


class FlightCondition:
    """Where the aircraft is and what it is doing, at one or many points along a mission.

    The quantities a loads model is entitled to ask about. Every one is published by
    OpenConcept's mission at every node under a ``fltcond|`` name, so a model can be evaluated
    over a whole phase at once.

    Parameters
    ----------
    CL : array_like
        Lift coefficient. Supplied *to* the model rather than computed by it: the trajectory
        solves vertical equilibrium and tells the aircraft what lift it is producing.
    mach : array_like
        Flight Mach number.
    altitude : array_like, optional
        Geometric altitude in metres. Default 0.
    dynamic_pressure : array_like, optional
        Dynamic pressure in pascals. Default 0. Needed to turn a coefficient into a force.

    Raises
    ------
    ValueError
        If the quantities are given at different numbers of points.
    """

    NAMES: ClassVar[tuple[str, ...]] = ("CL", "mach", "altitude", "dynamic_pressure")

    __slots__ = ("_values",)

    def __init__(
        self,
        CL: object,
        mach: object,
        altitude: object = 0.0,
        dynamic_pressure: object = 0.0,
    ) -> None:
        arrays = [np.atleast_1d(np.asarray(value, dtype=float)) for value in (CL, mach, altitude, dynamic_pressure)]
        nodes = max(array.size for array in arrays)
        for name, array in zip(self.NAMES, arrays, strict=True):
            if array.size not in (1, nodes):
                raise ValueError(f"'{name}' has {array.size} values but the flight condition has {nodes}.")
        self._values = {
            name: (np.full(nodes, float(array[0])) if array.size == 1 else array)
            for name, array in zip(self.NAMES, arrays, strict=True)
        }

    @property
    def CL(self) -> np.ndarray:
        """Lift coefficient the trajectory is asking the aircraft to produce."""
        return self._values["CL"]

    @property
    def mach(self) -> np.ndarray:
        """Flight Mach number."""
        return self._values["mach"]

    @property
    def altitude(self) -> np.ndarray:
        """Geometric altitude, metres."""
        return self._values["altitude"]

    @property
    def dynamic_pressure(self) -> np.ndarray:
        """Dynamic pressure, pascals."""
        return self._values["dynamic_pressure"]

    def __len__(self) -> int:
        """Return how many points this condition describes."""
        return int(self._values["CL"].size)

    def __repr__(self) -> str:
        """Return a representation naming the point count."""
        return f"FlightCondition({len(self)} points)"


class Planform(ABC):
    """The wing shape a loads model is evaluated on.

    Abstract because what a model needs from the geometry depends on the model. A parabolic polar
    needs area and aspect ratio; a vortex-lattice method needs the actual lifting surface. Both
    are described here by the same four numbers a case file declares, and it is the
    implementation's business to turn them into whatever it solves on.
    """

    @property
    @abstractmethod
    def area(self) -> float:
        """Reference wing area, square metres."""

    @property
    @abstractmethod
    def aspect_ratio(self) -> float:
        """Wing aspect ratio."""


class AerodynamicLoads(ABC):
    """A model that produces aerodynamic loads for a flight condition and a planform.

    Subclasses implement :meth:`build`, :meth:`coefficients` and :meth:`drag_gradients`. Everything
    else -- installing the model in an OpenMDAO group, vectorising it over a mission, handing its
    drag to the trajectory -- belongs to :mod:`cdadt.adapter` and is not a model's concern.

    Attributes
    ----------
    model_name : str
        Short identifier, used in reports and in the run record. Abstract in the same sense as a
        discipline's name: it is the one thing no model can inherit.
    """

    model_name: ClassVar[str]

    #: The names :meth:`drag_gradients` must answer for. ``CL`` and ``CD0`` are the flight-point
    #: quantities; the rest are the wing numbers a case file declares. A model that a variable
    #: genuinely does not reach still reports zero for it -- explicitly, so that "this does not
    #: affect my drag" is a statement the model makes rather than an omission the wrapper guesses.
    GRADIENT_NAMES: ClassVar[tuple[str, ...]] = ("CL", "CD0", "e", "area", "AR", "sweep", "taper")

    #: Which of :attr:`GRADIENT_NAMES` this model's drag coefficient can actually depend on. A
    #: subclass narrows it to say so structurally, which is what lets the wrapper declare a truthful
    #: sparsity pattern instead of claiming a dependence it then reports as zero. The default is
    #: everything, because over-declaring is merely wasteful whereas under-declaring is wrong.
    DRAG_DEPENDS_ON: ClassVar[tuple[str, ...]] = GRADIENT_NAMES

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require a subclass to name itself, at the moment the class is written.

        Raises
        ------
        TypeError
            If ``model_name`` was not declared on the subclass. Without it two models would be
            reported under the same name and a run record would not say which was used.
        """
        super().__init_subclass__(**kwargs)
        if "model_name" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare a 'model_name'; it is what a run record names as "
                f"the aerodynamics that produced its numbers."
            )

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        planform: Planform,
        span_efficiency: float,
        zero_lift_drag: object,
        workspace: object = None,
    ) -> AerodynamicLoads:
        """Construct this model from the black box's current values.

        Every model is offered the same three things and takes what it needs -- a parabolic polar
        wants the span efficiency and ignores the wing's shape; a vortex lattice wants the shape
        and computes the efficiency itself. Declaring that choice explicitly is why this is
        abstract rather than a shared constructor signature: what a model consumes is part of
        what it *is*.

        Called on every evaluation, with values as they currently stand, because the span
        efficiency and the zero-lift drag are variables of the black box rather than
        configuration. **Construction must therefore be cheap.**

        ``workspace`` is how a model that cannot be cheap stays cheap. A vortex lattice costs tens of
        seconds per geometry and must not pay that per Newton iteration, so it needs somewhere to
        keep solved results that outlives one evaluation. The obvious answer -- a module-level cache
        -- is a global by another name, and the brief forbids those; so the *caller* owns the store
        and injects it, which puts its lifetime where the decision belongs. The analysis group
        creates one per study and threads it down, so every phase of a mission shares one.

        A model must work with ``workspace=None``, which means "no store offered, solve what you
        need". A model that has nothing expensive to keep ignores the argument entirely --
        :class:`~cdadt.models.polar.PolarLoads` does. What the object *is* is the model's business:
        the interface promises only that it is handed back unchanged on every call.
        """

    @classmethod
    def new_workspace(cls) -> object | None:
        """Return a fresh store for whatever this model is expensive to recompute, or ``None``.

        The counterpart to the ``workspace`` argument of :meth:`build`. That argument settles *who
        owns the lifetime* -- the caller, so that one store serves a whole study rather than one
        evaluation. This settles *what the store is*, which is the model's business and nothing
        else's: a caller that had to know would have to know which solver sits underneath, and the
        whole point of the slot is that it does not.

        Default ``None``, for a model with nothing expensive to keep. A caller creates one of these
        per study and hands it back to :meth:`build` on every evaluation.

        This exists because the alternative failed in exactly the way that matters. The analysis
        group used to construct the store itself, which silently meant *openavl's* store -- so
        naming the OpenAeroStruct model in a case file handed it a library full of another code's
        answers for the same wing. The two differ by about 4%, so nothing would have raised had the
        library not refused it by name.
        """
        return None

    @abstractmethod
    def coefficients(self, condition: FlightCondition, planform: Planform) -> AeroCoefficients:
        """Return the aerodynamic coefficients at every point of ``condition``.

        Parameters
        ----------
        condition : FlightCondition
            Where the aircraft is and what lift it is producing.
        planform : Planform
            The wing being evaluated.

        Returns
        -------
        AeroCoefficients
            Six components, at the same number of points as ``condition``.
        """

    @abstractmethod
    def drag_gradients(self, condition: FlightCondition, planform: Planform) -> dict[str, np.ndarray]:
        """Return the derivatives of the drag coefficient, one entry per :attr:`GRADIENT_NAMES`.

        Abstract because an optimizer steps on these. A default implementation could only be a
        finite difference or a zero, and either would let a model ship with derivatives that are
        quietly not its own -- which is the failure this framework exists to avoid.

        Each value has the same shape as ``condition.CL``. The wrapper that installs the model
        applies the product rule for ``drag = CD q S`` itself, so what is wanted here is the
        derivative of the *coefficient* and nothing more.

        Returns
        -------
        dict
            Keyed by :attr:`GRADIENT_NAMES`; ``area``, ``AR``, ``sweep`` and ``taper`` are per unit
            of the wing number, in the units :class:`~cdadt.models.planform.TrapezoidalPlanform`
            uses -- square metres for area and degrees for sweep.
        """

    def drag(self, condition: FlightCondition, planform: Planform) -> np.ndarray:
        """Return the drag force in newtons, which is what the trajectory consumes.

        Not abstract: it is :meth:`coefficients` times dynamic pressure and area, and a model
        that overrode it could report a drag inconsistent with its own drag coefficient.
        """
        return self.coefficients(condition, planform).CD * condition.dynamic_pressure * planform.area

    def __repr__(self) -> str:
        """Return a representation naming the model."""
        return f"{type(self).__name__}({self.model_name!r})"
