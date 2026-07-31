"""The initial conditions a run starts from, and the ladder that converges the hard ones.

This module is the direct counterpart of ``set_values(prob, num_nodes)`` in OpenConcept's
``B738.py`` and of ``set_mission_profile(prob)`` in its ``B738_sizing.py``. Both are functions
that write a list of values into a problem before running it:

.. code-block:: python

   prob.set_val("climb.fltcond|vs", np.linspace(2300.0, 600.0, num_nodes), units="ft/min")
   prob.set_val("climb.fltcond|Ueas", np.linspace(230, 220, num_nodes), units="kn")
   prob.set_val("cruise|h0", 33000.0, units="ft")
   prob.set_val("mission_range", 2050, units="NM")

:class:`InitialConditions` is that list written as data instead of as statements, with the same
names. A value may be a single number, a pair of endpoints to interpolate across the phase
exactly as ``np.linspace`` does above, or one number per analysis node.

:class:`ContinuationStep` and :class:`ContinuationLadder` are the second half of
``set_mission_profile``: the part that converges an easy mission first and steps up to the
design one, because a Newton solver started cold on a 2800 nmi mission at 35,000 ft does not
reach it. OpenConcept writes that as two ``run_model()`` calls between blocks of assignments;
here it is a list of rungs, so it travels with the case it converges.

Nothing here imports OpenConcept or OpenMDAO. Conditions are written into anything that accepts
``set(name, value, units)``, ``run()`` and ``shape_of(name)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

__all__ = [
    "ContinuationLadder",
    "ContinuationStep",
    "InitialConditions",
    "MissionError",
    "SupportsConditions",
]


class MissionError(Exception):
    """Raised when an initial condition cannot be resampled onto the grid it is written to."""


class SupportsConditions(Protocol):
    """Anything initial conditions can be written into.

    Deliberately narrow: the profile needs to set a value, ask how big the target is so a pair
    of endpoints can be interpolated across it, and converge. That keeps this module free of
    any OpenMDAO import and testable without building a model.
    """

    def set(self, name: str, value: Any, units: str | None = None) -> None:
        """Set ``name`` to ``value``, interpreting it in ``units``."""
        ...

    def shape_of(self, name: str) -> tuple[int, ...]:
        """Return the shape the box declares for ``name``."""
        ...

    def run(self) -> None:
        """Converge the current state."""
        ...


class InitialConditions:
    """The values a run starts from, by name.

    Parameters
    ----------
    values : mapping
        ``name -> (value, units)``. The value may be a float, a two-element sequence of
        endpoints, or one value per analysis node.
    mission_path : str, optional
        Subsystem the mission lives under inside the black box. Default ``"mission"``.

        Names are written exactly as OpenConcept's own run scripts write them --
        ``cruise|h0``, ``climb.fltcond|vs`` -- because in those scripts the mission is promoted
        to the top of the model. In the sizing analysis it is a subsystem, so a name that is not
        settable on its own is retried under this path. A name that resolves either way resolves
        the same way every time, and :meth:`resolve` is what a message quotes when it does not
        resolve at all.

    Examples
    --------
    >>> conditions = InitialConditions({"cruise|h0": (35000.0, "ft")})
    >>> conditions.resolve("cruise|h0", box)
    'mission.cruise|h0'
    """

    def __init__(self, values: Mapping[str, tuple[Any, str | None]], mission_path: str = "mission") -> None:
        self._values = dict(values)
        self._mission_path = str(mission_path)

    @property
    def mission_path(self) -> str:
        """Subsystem the mission lives under inside the black box."""
        return self._mission_path

    @property
    def values(self) -> dict[str, tuple[Any, str | None]]:
        """Return the conditions as ``name -> (value, units)``."""
        return dict(self._values)

    def __contains__(self, name: str) -> bool:
        """Return whether a condition of that name is set."""
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        """Iterate the condition names, in the order the case file wrote them."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return how many conditions are set."""
        return len(self._values)

    def merged_with(self, other: InitialConditions | Mapping[str, tuple[Any, str | None]]) -> InitialConditions:
        """Return these conditions with ``other`` written on top.

        How a continuation rung is applied: the design conditions first, then whatever the rung
        overrides, so a rung states only what it relaxes.
        """
        overrides = other.values if isinstance(other, InitialConditions) else dict(other)
        return InitialConditions({**self._values, **overrides}, self._mission_path)

    # -- resolving and writing ------------------------------------------------------------

    def resolve(self, name: str, box: Any) -> str:
        """Return the name the black box actually publishes this condition under.

        Tries the name as written, then under the mission path. Raises naming both attempts,
        because "is it ``cruise|h0`` or ``mission.cruise|h0``" is the first question a failure
        here raises and the message should answer it.
        """
        if box.has(name):
            return name
        prefixed = f"{self._mission_path}.{name}"
        if box.has(prefixed):
            return prefixed
        raise MissionError(
            f"The black box publishes neither '{name}' nor '{prefixed}'. "
            f"Run 'cdadt inspect <case>' to list what it does publish."
        )

    def resample(self, value: Any, shape: tuple[int, ...], name: str) -> Any:
        """Return ``value`` shaped to fit ``shape``.

        A float is left alone and broadcast by OpenMDAO. A two-element sequence is interpolated
        across the target, which is what ``np.linspace(2300.0, 600.0, num_nodes)`` does in
        OpenConcept's own run script. A full-length sequence is used as given.

        Raises
        ------
        MissionError
            For any other length. Broadcasting a wrong-length schedule would quietly fly a
            different mission than the one the case file asks for.
        """
        if not isinstance(value, list | tuple | np.ndarray):
            return float(value)
        array = np.atleast_1d(np.asarray(value, dtype=float))
        size = int(np.prod(shape)) if shape else 1
        if array.size == size:
            return array
        if array.size == 1:
            return np.full(size, array[0])
        if array.size == 2 and size > 2:
            return np.linspace(array[0], array[1], size)
        raise MissionError(
            f"'{name}' was given {array.size} values but the black box declares {size}. "
            f"Give {size} values, 2 endpoints to interpolate between, or 1 constant."
        )

    def apply(self, box: SupportsConditions) -> None:
        """Write every condition into the box, without converging it."""
        for name, (value, units) in self._values.items():
            resolved = self.resolve(name, box)
            box.set(resolved, self.resample(value, box.shape_of(resolved), name), units=units)

    def check(self, box: Any) -> None:
        """Raise if any condition names something the box does not publish."""
        for name in self._values:
            self.resolve(name, box)

    def __repr__(self) -> str:
        """Return a representation naming how many conditions are held."""
        return f"InitialConditions({len(self._values)} values)"


class ContinuationStep:
    """One rung of the ladder up to the design mission.

    Parameters
    ----------
    description : str
        What this rung relaxes, for the run log.
    conditions : mapping, optional
        Initial-condition overrides for this rung, as ``name -> (value, units)``. A rung states
        only what it changes; everything else is the design condition.
    """

    __slots__ = ("_conditions", "_description")

    def __init__(self, description: str, conditions: Mapping[str, tuple[Any, str | None]] | None = None) -> None:
        self._description = str(description)
        self._conditions = dict(conditions or {})

    @property
    def description(self) -> str:
        """What this rung relaxes."""
        return self._description

    @property
    def conditions(self) -> dict[str, tuple[Any, str | None]]:
        """The overrides this rung applies."""
        return dict(self._conditions)

    def __repr__(self) -> str:
        """Return a representation naming the rung."""
        return f"ContinuationStep({self._description!r})"


class ContinuationLadder:
    """The rungs walked before the design mission, and the walk itself.

    Parameters
    ----------
    steps : sequence of ContinuationStep, optional
        Progressively harder missions, converged in order. Empty attempts the design mission
        directly, which for a long-range mission generally fails.
    """

    def __init__(self, steps: Sequence[ContinuationStep] = ()) -> None:
        self._steps = tuple(steps)

    @property
    def steps(self) -> tuple[ContinuationStep, ...]:
        """The rungs, in the order they are walked."""
        return self._steps

    def __len__(self) -> int:
        """Return how many rungs there are."""
        return len(self._steps)

    def __iter__(self) -> Iterator[ContinuationStep]:
        """Iterate the rungs."""
        return iter(self._steps)

    def converge(self, box: SupportsConditions, conditions: InitialConditions, verbose: bool = False) -> None:
        """Walk every rung, then converge the design conditions.

        Each rung is written on top of the design conditions and converged, so the solver enters
        the next rung from a converged neighbour. The final run is the design mission itself.
        """
        for step in self._steps:
            if verbose:
                print(f"[cdadt] continuation: {step.description}")
            conditions.merged_with(step.conditions).apply(box)
            box.run()

        if verbose:
            print("[cdadt] continuation complete; converging the design mission")
        conditions.apply(box)
        box.run()

    def __repr__(self) -> str:
        """Return a representation naming how many rungs there are."""
        return f"ContinuationLadder({len(self._steps)} steps)"
