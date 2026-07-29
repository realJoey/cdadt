"""The six aerodynamic coefficients, and the names they are published under.

A loads model produces forces and moments in coefficient form. Six numbers describe them
completely in the aircraft's own axes, and every one is something a vortex-lattice method
computes whether or not the caller asked: there is no saving in returning fewer.

The mission cdadt drives consumes only the drag today, because OpenConcept's trajectory solves
vertical equilibrium itself and asks the aircraft model for drag alone. The other five are carried anyway. Trim, static margin, control authority and the
handling-qualities side of a certification basis are all written in terms of the moment
coefficients, and a loads interface that discarded them would have to be widened later by
changing every implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import numpy as np

__all__ = ["AeroCoefficients"]


class AeroCoefficients:
    """Lift, drag, sideforce and the three moment coefficients, at one or many flight points.

    Every component is stored as an array so that a model evaluated over a whole mission phase
    and a model evaluated at a single point are the same object with different lengths.

    Parameters
    ----------
    CL, CD : array_like
        Lift and drag coefficients. Required: no aerodynamic model produces neither.
    CY, Cl, Cm, Cn : array_like, optional
        Sideforce, rolling-, pitching- and yawing-moment coefficients. Default zero, which is
        what a symmetric model in symmetric flight correctly reports -- as distinct from "not
        computed", which :attr:`lateral_directional` distinguishes.

    Raises
    ------
    ValueError
        If the components do not all have the same length. A loads model that returned drag at
        every node and a moment at one has a bug that would otherwise surface as a broadcast
        deep inside a solver.

    Examples
    --------
    >>> coefficients = AeroCoefficients(CL=[0.5, 0.6], CD=[0.02, 0.024])
    >>> coefficients.CD
    array([0.02 , 0.024])
    >>> len(coefficients)
    2
    >>> coefficients.lateral_directional
    False
    """

    #: The components, in the order they are reported.
    NAMES: ClassVar[tuple[str, ...]] = ("CL", "CD", "CY", "Cl", "Cm", "Cn")

    __slots__ = ("_values",)

    def __init__(
        self,
        CL: object,
        CD: object,
        CY: object = 0.0,
        Cl: object = 0.0,
        Cm: object = 0.0,
        Cn: object = 0.0,
    ) -> None:
        given = (CL, CD, CY, Cl, Cm, Cn)
        arrays = [np.atleast_1d(np.asarray(value, dtype=float)) for value in given]

        nodes = max(array.size for array in arrays)
        for name, array in zip(self.NAMES, arrays, strict=True):
            if array.size not in (1, nodes):
                raise ValueError(
                    f"'{name}' has {array.size} values but the coefficients have {nodes}. Every "
                    f"component must be given at the same points, or as a single value."
                )
        self._values = {
            name: (np.full(nodes, float(array[0])) if array.size == 1 else array)
            for name, array in zip(self.NAMES, arrays, strict=True)
        }

    # -- the components ------------------------------------------------------------------

    @property
    def CL(self) -> np.ndarray:
        """Lift coefficient."""
        return self._values["CL"]

    @property
    def CD(self) -> np.ndarray:
        """Drag coefficient. The one the mission consumes."""
        return self._values["CD"]

    @property
    def CY(self) -> np.ndarray:
        """Sideforce coefficient."""
        return self._values["CY"]

    @property
    def Cl(self) -> np.ndarray:
        """Rolling-moment coefficient."""
        return self._values["Cl"]

    @property
    def Cm(self) -> np.ndarray:
        """Pitching-moment coefficient."""
        return self._values["Cm"]

    @property
    def Cn(self) -> np.ndarray:
        """Yawing-moment coefficient."""
        return self._values["Cn"]

    # -- what they describe --------------------------------------------------------------

    @property
    def lateral_directional(self) -> bool:
        """Whether anything out of the symmetric plane is non-zero.

        A symmetric aircraft in symmetric flight genuinely has zero sideforce and zero rolling
        and yawing moment, so zero is an answer rather than an absence. This says which case a
        reader is looking at without having to compare six arrays by eye.
        """
        return any(np.any(self._values[name] != 0.0) for name in ("CY", "Cl", "Cn"))

    def __getitem__(self, name: str) -> np.ndarray:
        """Return one component by name, so a caller may select one from configuration."""
        if name not in self._values:
            raise KeyError(f"'{name}' is not an aerodynamic coefficient; expected one of {list(self.NAMES)}.")
        return self._values[name]

    def __len__(self) -> int:
        """Return how many flight points these coefficients describe."""
        return int(self._values["CL"].size)

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        """Iterate ``(name, values)`` in report order."""
        return iter((name, self._values[name]) for name in self.NAMES)

    def __repr__(self) -> str:
        """Return a representation naming the point count and whether it is symmetric."""
        kind = "6-component" if self.lateral_directional else "symmetric"
        return f"AeroCoefficients({len(self)} points, {kind})"
