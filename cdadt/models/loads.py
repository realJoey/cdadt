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

    #: The case-file variables every planform is built from.
    VARIABLES: ClassVar[tuple[str, ...]] = (
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|taper",
    )

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

    Subclasses implement :meth:`coefficients` and declare :attr:`requires`. Everything else --
    installing the model in an OpenMDAO group, vectorising it over a mission, handing its drag to
    the trajectory -- belongs to :mod:`cdadt.adapter` and is not a model's concern.

    Attributes
    ----------
    model_name : str
        Short identifier, used in reports and in the run record. Abstract in the same sense as a
        discipline's name: it is the one thing no model can inherit.
    requires : tuple of str
        Black-box variables this model reads. Declared so that a case file naming a model can be
        checked against the box *before* a long run starts, rather than failing inside setup.
    """

    model_name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

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

    def drag(self, condition: FlightCondition, planform: Planform) -> np.ndarray:
        """Return the drag force in newtons, which is what the trajectory consumes.

        Not abstract: it is :meth:`coefficients` times dynamic pressure and area, and a model
        that overrode it could report a drag inconsistent with its own drag coefficient.
        """
        return self.coefficients(condition, planform).CD * condition.dynamic_pressure * planform.area

    def __repr__(self) -> str:
        """Return a representation naming the model."""
        return f"{type(self).__name__}({self.model_name!r})"
