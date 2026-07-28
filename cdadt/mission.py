"""The mission the aircraft is sized against, and the ladder that converges it.

Three value objects and no analysis. :class:`PhaseSchedule` is the speed and vertical speed
flown in one phase, :class:`ContinuationStep` is one rung of an easier-mission ladder, and
:class:`MissionProfile` is the whole thing: the design mission, the per-phase schedules, and
the continuation sequence that reaches them.

Continuation is part of the interface, not a convenience. The black box is a coupled implicit
system -- phase durations are solved against altitude and range targets, throttle against zero
acceleration, and the decision speed V\\ :sub:`1` so that the continue and abort distances
match. A Newton solver started cold on a 2800 nmi mission at 35,000 ft does not generally reach
it. OpenConcept's own sizing example handles this by converging an easy mission first and
stepping up, written as a sequence of bare assignments in a run script. Here the same idea is
data: the steps are declared in the case file, can be inspected, and travel with the mission
they converge.

Nothing in this module imports OpenConcept or OpenMDAO. A profile is applied to an object that
can be told to set a value -- see :class:`~cdadt.blackbox.OpenConceptSizingBox`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

__all__ = [
    "GROUND_ROLL_PHASES",
    "STEADY_FLIGHT_PHASES",
    "ContinuationStep",
    "MissionProfile",
    "PhaseSchedule",
    "SupportsSet",
]

#: Phases the black box integrates along a speed and vertical-speed schedule. Every one of them
#: must be scheduled: an unscheduled phase flies whatever placeholder its component declared,
#: which is a different mission than the one the case file asks for.
STEADY_FLIGHT_PHASES: tuple[str, ...] = (
    "climb",
    "cruise",
    "descent",
    "reserve_climb",
    "reserve_cruise",
    "reserve_descent",
    "loiter",
)

#: Ground-roll phases. They integrate acceleration from a standstill, so their solver needs a
#: true-airspeed starting guess rather than a schedule.
GROUND_ROLL_PHASES: tuple[str, ...] = ("v0v1", "v1vr", "v1v0")


class SupportsSet(Protocol):
    """Anything a mission profile can be written into.

    The profile does not need a black box, only something that accepts a value at a path. That
    keeps this module free of any OpenMDAO or OpenConcept import, and makes the profile
    testable without building a model.
    """

    def set(self, name: str, value: Any, units: str | None = None) -> None:
        """Set ``name`` to ``value``, interpreting it in ``units``."""
        ...

    def run(self) -> None:
        """Converge the current state."""
        ...


class PhaseSchedule:
    """The speed and vertical speed flown in one steady-flight phase.

    Parameters
    ----------
    equivalent_airspeed : sequence of float or float
        Equivalent airspeed. Either one value per analysis node, or exactly two endpoints to
        interpolate linearly between, or a single constant.
    vertical_speed : sequence of float or float
        Vertical speed, same length convention.
    airspeed_units : str, optional
        Default ``"kn"``.
    vertical_speed_units : str, optional
        Default ``"ft/min"``.

    Examples
    --------
    >>> climb = PhaseSchedule([230, 252], [2300, 400])
    >>> climb.airspeed_at(5)
    array([230., 235.5, 241., 246.5, 252.])
    """

    __slots__ = ("_airspeed_units", "_equivalent_airspeed", "_vertical_speed", "_vertical_speed_units")

    def __init__(
        self,
        equivalent_airspeed: Sequence[float] | float,
        vertical_speed: Sequence[float] | float,
        airspeed_units: str = "kn",
        vertical_speed_units: str = "ft/min",
    ) -> None:
        self._equivalent_airspeed = np.atleast_1d(np.asarray(equivalent_airspeed, dtype=float))
        self._vertical_speed = np.atleast_1d(np.asarray(vertical_speed, dtype=float))
        self._airspeed_units = airspeed_units
        self._vertical_speed_units = vertical_speed_units

    @property
    def airspeed_units(self) -> str:
        """Units the airspeed schedule is expressed in."""
        return self._airspeed_units

    @property
    def vertical_speed_units(self) -> str:
        """Units the vertical-speed schedule is expressed in."""
        return self._vertical_speed_units

    def airspeed_at(self, num_nodes: int) -> np.ndarray:
        """Return the equivalent-airspeed schedule as a length-``num_nodes`` array."""
        return self._resample(self._equivalent_airspeed, num_nodes, "equivalent airspeed")

    def vertical_speed_at(self, num_nodes: int) -> np.ndarray:
        """Return the vertical-speed schedule as a length-``num_nodes`` array."""
        return self._resample(self._vertical_speed, num_nodes, "vertical speed")

    @staticmethod
    def _resample(values: np.ndarray, num_nodes: int, label: str) -> np.ndarray:
        """Return ``values`` at ``num_nodes`` points.

        Raises
        ------
        ValueError
            If the schedule is neither the right length, a pair of endpoints, nor a constant.
            Broadcasting a wrong-length schedule would quietly fly a different mission than the
            one the case file asks for.
        """
        if values.size == num_nodes:
            return values.copy()
        if values.size == 2:
            return np.linspace(values[0], values[1], num_nodes)
        if values.size == 1:
            return np.full(num_nodes, values[0])
        raise ValueError(
            f"The {label} schedule has {values.size} values but the mission has {num_nodes} nodes per phase. "
            f"Give {num_nodes} values, 2 endpoints to interpolate between, or 1 constant."
        )

    def __repr__(self) -> str:
        """Return a representation naming both schedules."""
        return (
            f"PhaseSchedule(Ueas={self._equivalent_airspeed.tolist()} {self._airspeed_units}, "
            f"vs={self._vertical_speed.tolist()} {self._vertical_speed_units})"
        )


class ContinuationStep:
    """One rung of the ladder up to the design mission.

    Parameters
    ----------
    description : str
        What this step relaxes, for the run log.
    parameters : mapping, optional
        Mission-parameter overrides for this step, as ``name -> (value, units)``.
    schedules : mapping, optional
        Per-phase schedule overrides for this step.
    """

    __slots__ = ("_description", "_parameters", "_schedules")

    def __init__(
        self,
        description: str,
        parameters: Mapping[str, tuple[float, str | None]] | None = None,
        schedules: Mapping[str, PhaseSchedule] | None = None,
    ) -> None:
        self._description = str(description)
        self._parameters = dict(parameters or {})
        self._schedules = dict(schedules or {})

    @property
    def description(self) -> str:
        """What this step relaxes."""
        return self._description

    @property
    def parameters(self) -> Mapping[str, tuple[float, str | None]]:
        """Mission-parameter overrides applied by this step."""
        return dict(self._parameters)

    @property
    def schedules(self) -> Mapping[str, PhaseSchedule]:
        """Per-phase schedule overrides applied by this step."""
        return dict(self._schedules)

    def __repr__(self) -> str:
        """Return a representation naming the step."""
        return f"ContinuationStep({self._description!r})"


class MissionProfile:
    """The mission to fly and the continuation ladder that reaches it.

    Parameters
    ----------
    parameters : mapping
        Mission-level values as ``name -> (value, units)``. Names are the black box's own:
        ``mission_range``, ``reserve_range``, ``cruise|h0``, ``reserve|h0``, ``loiter|h0``,
        ``loiter_duration``, ``takeoff|h``.
    schedules : mapping
        One :class:`PhaseSchedule` per phase in :data:`STEADY_FLIGHT_PHASES`.
    continuation : sequence of ContinuationStep, optional
        Progressively harder missions, run in order before the design mission. Empty means the
        design mission is attempted directly, which for a long-range mission generally fails.
    takeoff_speed_guess : tuple of (float, str), optional
        Starting guess for the ground-roll true-airspeed vectors. Default ``(100.0, "kn")``.
    mission_path : str, optional
        Name of the mission subsystem inside the black box. Default ``"mission"``.

    Raises
    ------
    ValueError
        If a steady-flight phase has no schedule.
    """

    __slots__ = ("_continuation", "_mission_path", "_parameters", "_schedules", "_takeoff_speed_guess")

    def __init__(
        self,
        parameters: Mapping[str, tuple[float, str | None]],
        schedules: Mapping[str, PhaseSchedule],
        continuation: Sequence[ContinuationStep] = (),
        takeoff_speed_guess: tuple[float, str] = (100.0, "kn"),
        mission_path: str = "mission",
    ) -> None:
        missing = [phase for phase in STEADY_FLIGHT_PHASES if phase not in schedules]
        if missing:
            raise ValueError(
                f"No airspeed/vertical-speed schedule for {missing}. Every steady-flight phase must be "
                f"scheduled, because an unscheduled phase flies its component's placeholder values."
            )
        self._parameters = dict(parameters)
        self._schedules = dict(schedules)
        self._continuation = tuple(continuation)
        self._takeoff_speed_guess = (float(takeoff_speed_guess[0]), takeoff_speed_guess[1])
        self._mission_path = str(mission_path)

    # -- state ---------------------------------------------------------------------------

    @property
    def parameters(self) -> Mapping[str, tuple[float, str | None]]:
        """Mission-level parameters of the design mission."""
        return dict(self._parameters)

    @property
    def schedules(self) -> Mapping[str, PhaseSchedule]:
        """Per-phase speed and vertical-speed schedules of the design mission."""
        return dict(self._schedules)

    @property
    def continuation(self) -> tuple[ContinuationStep, ...]:
        """The ladder of easier missions run before the design mission."""
        return self._continuation

    @property
    def takeoff_speed_guess(self) -> tuple[float, str]:
        """Starting guess for the ground-roll true-airspeed vectors, as ``(value, units)``."""
        return self._takeoff_speed_guess

    @property
    def mission_path(self) -> str:
        """Name of the mission subsystem inside the black box."""
        return self._mission_path

    def __iter__(self) -> Iterator[str]:
        """Iterate the scheduled phase names."""
        return iter(self._schedules)

    # -- applying ------------------------------------------------------------------------

    def apply(self, box: SupportsSet, num_nodes: int) -> None:
        """Write the design mission into ``box`` without converging it.

        Parameters
        ----------
        box : SupportsSet
            The black box, or anything that accepts ``set(name, value, units)``.
        num_nodes : int
            Analysis points per phase, used to resample the schedules.
        """
        self._write_parameters(box, self._parameters)
        self._write_schedules(box, self._schedules, num_nodes)

    def seed_takeoff_speeds(self, box: SupportsSet, num_nodes: int) -> None:
        """Seed the ground-roll true-airspeed vectors.

        These phases integrate acceleration from a standstill and their solver is sensitive to
        where it starts. Seeding happens once, before the first solve; re-seeding between
        continuation steps would discard the converged state each step exists to provide.
        """
        value, units = self._takeoff_speed_guess
        for phase in GROUND_ROLL_PHASES:
            box.set(f"{self._mission_path}.{phase}.fltcond|Utrue", np.full(num_nodes, value), units=units)

    def converge(self, box: SupportsSet, num_nodes: int, verbose: bool = False) -> None:
        """Walk the continuation ladder, then converge the design mission.

        Each step is written on top of the design profile and the box run, so the solver enters
        the next step from a converged neighbour. The final run is the design mission itself.

        Parameters
        ----------
        box : SupportsSet
            The black box.
        num_nodes : int
            Analysis points per phase.
        verbose : bool, optional
            Print each step as it runs. Default ``False``.
        """
        self.seed_takeoff_speeds(box, num_nodes)
        for step in self._continuation:
            if verbose:
                print(f"[cdadt] continuation: {step.description}")
            self.apply(box, num_nodes)
            self._write_parameters(box, step.parameters)
            self._write_schedules(box, step.schedules, num_nodes)
            box.run()

        if verbose:
            print("[cdadt] continuation complete; converging the design mission")
        self.apply(box, num_nodes)
        box.run()

    # -- internals -----------------------------------------------------------------------

    def _write_parameters(self, box: SupportsSet, parameters: Mapping[str, tuple[float, str | None]]) -> None:
        """Write mission-level parameters into the box."""
        for name, (value, units) in parameters.items():
            box.set(f"{self._mission_path}.{name}", value, units=units)

    def _write_schedules(self, box: SupportsSet, schedules: Mapping[str, PhaseSchedule], num_nodes: int) -> None:
        """Write the airspeed and vertical-speed vectors of the given phases into the box."""
        for phase, schedule in schedules.items():
            box.set(
                f"{self._mission_path}.{phase}.fltcond|Ueas",
                schedule.airspeed_at(num_nodes),
                units=schedule.airspeed_units,
            )
            box.set(
                f"{self._mission_path}.{phase}.fltcond|vs",
                schedule.vertical_speed_at(num_nodes),
                units=schedule.vertical_speed_units,
            )

    def __repr__(self) -> str:
        """Return a representation naming the phase and continuation-step counts."""
        return f"MissionProfile({len(self._schedules)} phases, {len(self._continuation)} continuation steps)"
