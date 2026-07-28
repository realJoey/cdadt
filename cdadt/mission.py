"""The mission profile: what the aircraft flies, and how the solver is walked up to it.

Two things live here, and the second is the reason this module exists at all.

The first is the profile: the design range, the cruise altitude, the reserve mission, and the
airspeed and vertical-speed schedule flown in each phase.

The second is the **continuation schedule**. OpenConcept's mission is a coupled implicit
system -- phase durations are solved against altitude and range targets, throttle against zero
acceleration, and V\\ :sub:`1` so that the continue and abort distances match. A Newton solver
started on a 2800 nmi mission at 35,000 ft does not generally converge to it from a cold
start. OpenConcept's own example handles this by running an easy mission first and stepping up,
and it does so as a sequence of bare ``set_val`` calls in a run script. Here the same idea is
a declared object: the steps can be inspected, changed per case, and written in a YAML file
alongside the mission they converge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import openmdao.api as om
import yaml

__all__ = ["ContinuationStep", "MissionProfile", "PhaseSchedule"]

#: Phases OpenConcept integrates along a speed and vertical-speed schedule.
STEADY_FLIGHT_PHASES: tuple[str, ...] = (
    "climb",
    "cruise",
    "descent",
    "reserve_climb",
    "reserve_cruise",
    "reserve_descent",
    "loiter",
)

#: Ground-roll phases, which integrate acceleration from a standstill and need a true-airspeed
#: starting guess.
GROUND_ROLL_PHASES: tuple[str, ...] = ("v0v1", "v1vr", "v1v0")


@dataclass(frozen=True)
class PhaseSchedule:
    """The speed and vertical speed flown in one steady-flight phase.

    Parameters
    ----------
    equivalent_airspeed : sequence of float
        Equivalent airspeed at each analysis node. Either exactly ``num_nodes`` values, or
        exactly two endpoints to interpolate linearly between.
    vertical_speed : sequence of float
        Vertical speed, same length convention.
    airspeed_units : str, optional
        Default ``"kn"``.
    vertical_speed_units : str, optional
        Default ``"ft/min"``.
    """

    equivalent_airspeed: Sequence[float]
    vertical_speed: Sequence[float]
    airspeed_units: str = "kn"
    vertical_speed_units: str = "ft/min"

    def airspeed_at(self, num_nodes: int) -> np.ndarray:
        """Return the airspeed schedule as a length-``num_nodes`` array."""
        return self._resample(self.equivalent_airspeed, num_nodes, "equivalent_airspeed")

    def vertical_speed_at(self, num_nodes: int) -> np.ndarray:
        """Return the vertical-speed schedule as a length-``num_nodes`` array."""
        return self._resample(self.vertical_speed, num_nodes, "vertical_speed")

    @staticmethod
    def _resample(values: Sequence[float], num_nodes: int, label: str) -> np.ndarray:
        """Return ``values`` at ``num_nodes`` points, interpolating a two-point schedule.

        Raises
        ------
        ValueError
            If the schedule is neither the right length nor a pair of endpoints. Broadcasting
            a wrong-length schedule would quietly fly a different mission than the one asked
            for.
        """
        array = np.atleast_1d(np.asarray(values, dtype=float))
        if array.size == num_nodes:
            return array
        if array.size == 2:
            return np.linspace(array[0], array[1], num_nodes)
        if array.size == 1:
            return np.full(num_nodes, array[0])
        raise ValueError(
            f"{label} has {array.size} values but the mission has {num_nodes} nodes per phase. "
            f"Give {num_nodes} values, 2 endpoints to interpolate, or 1 constant."
        )


@dataclass(frozen=True)
class ContinuationStep:
    """One rung of the ladder up to the design mission.

    Parameters
    ----------
    description : str
        What this step relaxes, for the run log.
    parameters : mapping, optional
        Mission parameter overrides for this step, as ``name -> (value, units)``.
    schedules : mapping, optional
        Per-phase schedule overrides for this step.
    """

    description: str
    parameters: Mapping[str, tuple[float, str | None]] = field(default_factory=dict)
    schedules: Mapping[str, PhaseSchedule] = field(default_factory=dict)


@dataclass(frozen=True)
class MissionProfile:
    """The mission to fly and the continuation schedule that reaches it.

    Parameters
    ----------
    parameters : mapping
        Mission-level values as ``name -> (value, units)``. Names are OpenConcept's own, set
        on the mission group: ``mission_range``, ``reserve_range``, ``cruise|h0``,
        ``reserve|h0``, ``loiter|h0``, ``loiter_duration``, ``takeoff|h``.
    schedules : mapping
        One :class:`PhaseSchedule` per steady-flight phase. Every phase in
        :data:`STEADY_FLIGHT_PHASES` must be present; OpenConcept has no default profile to
        fall back on, and an unset phase flies whatever placeholder its component declared.
    continuation : sequence of ContinuationStep, optional
        Progressively harder missions, run in order before the design mission. Empty means
        the design mission is attempted directly.
    takeoff_speed_guess : tuple of (float, str), optional
        Starting guess for the ground-roll true-airspeed vectors. Default ``(100.0, "kn")``.

    Raises
    ------
    ValueError
        If a steady-flight phase has no schedule.
    """

    parameters: Mapping[str, tuple[float, str | None]]
    schedules: Mapping[str, PhaseSchedule]
    continuation: Sequence[ContinuationStep] = ()
    takeoff_speed_guess: tuple[float, str] = (100.0, "kn")

    def __post_init__(self) -> None:
        missing = [phase for phase in STEADY_FLIGHT_PHASES if phase not in self.schedules]
        if missing:
            raise ValueError(
                f"No airspeed/vertical-speed schedule for {missing}. Every steady-flight phase must be scheduled."
            )

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> MissionProfile:
        """Load a mission profile from a YAML file. See :meth:`from_dict` for the layout."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MissionProfile:
        """Build a profile from a nested mapping.

        Expected layout::

            parameters:
              mission_range: {value: 2800, units: nmi}
              cruise|h0:     {value: 35000, units: ft}
            schedule:
              climb:
                Ueas: {value: [230, 252], units: kn}
                vs:   {value: [2300, 400], units: ft/min}
            takeoff_speed_guess: {value: 100, units: kn}
            continuation:
              - description: short range at low altitude
                parameters: {mission_range: {value: 500, units: nmi}}
                schedule:
                  descent: {Ueas: {value: [252, 250], units: kn}, vs: {value: -800, units: ft/min}}

        Parameters
        ----------
        data : mapping
            The layout above.

        Returns
        -------
        MissionProfile
            The mission.
        """
        guess = data.get("takeoff_speed_guess", {"value": 100.0, "units": "kn"})
        return cls(
            parameters=cls._read_parameters(data.get("parameters", {})),
            schedules=cls._read_schedules(data.get("schedule", {})),
            continuation=tuple(
                ContinuationStep(
                    description=str(step.get("description", f"step {index}")),
                    parameters=cls._read_parameters(step.get("parameters", {})),
                    schedules=cls._read_schedules(step.get("schedule", {})),
                )
                for index, step in enumerate(data.get("continuation", []))
            ),
            takeoff_speed_guess=(float(guess["value"]), guess.get("units", "kn")),
        )

    @staticmethod
    def _read_parameters(entries: Mapping[str, Any]) -> dict[str, tuple[float, str | None]]:
        """Return ``name -> (value, units)`` from a mapping of ``{value, units}`` entries."""
        return {name: (float(entry["value"]), entry.get("units")) for name, entry in entries.items()}

    @staticmethod
    def _read_schedules(entries: Mapping[str, Any]) -> dict[str, PhaseSchedule]:
        """Return ``phase -> PhaseSchedule`` from a mapping of phase entries.

        Raises
        ------
        ValueError
            If a phase gives one of ``Ueas``/``vs`` without the other. Overriding one alone
            silently keeps the other from whatever was set before, which in a continuation
            step means flying a profile nobody wrote down.
        """
        schedules = {}
        for phase, entry in entries.items():
            if set(entry) != {"Ueas", "vs"}:
                raise ValueError(f"Phase '{phase}' gives {sorted(entry)}; both 'Ueas' and 'vs' are required.")
            schedules[phase] = PhaseSchedule(
                equivalent_airspeed=np.atleast_1d(entry["Ueas"]["value"]).tolist(),
                vertical_speed=np.atleast_1d(entry["vs"]["value"]).tolist(),
                airspeed_units=entry["Ueas"].get("units", "kn"),
                vertical_speed_units=entry["vs"].get("units", "ft/min"),
            )
        return schedules

    # -- applying -----------------------------------------------------------------------

    def apply(self, problem: om.Problem, num_nodes: int, mission_path: str = "mission") -> None:
        """Set the design mission on a problem, without running it.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem that has been set up and contains the mission.
        num_nodes : int
            Analysis points per phase, used to resample the schedules.
        mission_path : str, optional
            Path of the mission group. Default ``"mission"``.
        """
        self._apply_parameters(problem, self.parameters, mission_path)
        self._apply_schedules(problem, self.schedules, num_nodes, mission_path)

    def seed_takeoff_speeds(self, problem: om.Problem, num_nodes: int, mission_path: str = "mission") -> None:
        """Seed the ground-roll true-airspeed vectors.

        These phases integrate acceleration from a standstill and their solver is sensitive to
        where it starts. Seeding happens once, before the first solve; re-seeding between
        continuation steps would throw away the converged state each step exists to provide.
        """
        value, units = self.takeoff_speed_guess
        for phase in GROUND_ROLL_PHASES:
            problem.set_val(f"{mission_path}.{phase}.fltcond|Utrue", np.full(num_nodes, value), units=units)

    def converge(
        self,
        problem: om.Problem,
        num_nodes: int,
        mission_path: str = "mission",
        verbose: bool = False,
    ) -> None:
        """Walk the continuation schedule, then run the design mission.

        Each step is applied on top of the design profile and the model run, so the solver
        enters the next step from a converged neighbour. The final run is the design mission
        itself.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem that has been set up and contains the mission.
        num_nodes : int
            Analysis points per phase.
        mission_path : str, optional
            Path of the mission group. Default ``"mission"``.
        verbose : bool, optional
            Print each step as it runs. Default ``False``.
        """
        self.seed_takeoff_speeds(problem, num_nodes, mission_path)
        for step in self.continuation:
            if verbose:
                print(f"[cdadt] continuation: {step.description}")
            self.apply(problem, num_nodes, mission_path)
            self._apply_parameters(problem, step.parameters, mission_path)
            self._apply_schedules(problem, step.schedules, num_nodes, mission_path)
            problem.run_model()

        if verbose:
            print("[cdadt] continuation complete; running the design mission")
        self.apply(problem, num_nodes, mission_path)
        problem.run_model()

    @staticmethod
    def _apply_parameters(
        problem: om.Problem,
        parameters: Mapping[str, tuple[float, str | None]],
        mission_path: str,
    ) -> None:
        """Set mission-level parameters."""
        for name, (value, units) in parameters.items():
            problem.set_val(f"{mission_path}.{name}", value, units=units)

    @staticmethod
    def _apply_schedules(
        problem: om.Problem,
        schedules: Mapping[str, PhaseSchedule],
        num_nodes: int,
        mission_path: str,
    ) -> None:
        """Set the airspeed and vertical-speed vectors of the given phases."""
        for phase, schedule in schedules.items():
            problem.set_val(
                f"{mission_path}.{phase}.fltcond|Ueas",
                schedule.airspeed_at(num_nodes),
                units=schedule.airspeed_units,
            )
            problem.set_val(
                f"{mission_path}.{phase}.fltcond|vs",
                schedule.vertical_speed_at(num_nodes),
                units=schedule.vertical_speed_units,
            )
