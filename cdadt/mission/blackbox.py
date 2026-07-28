"""The mission black box: the single boundary between cdadt and OpenConcept.

This is the only module in cdadt that imports from ``openconcept.mission``. That is enforced
by :mod:`cdadt.tests.mission.test_boundary`, which scans the package for imports, so the
boundary cannot erode one convenience import at a time.

What "black box" means here
---------------------------
cdadt does not reimplement, modify, or reach into OpenConcept's mission physics. It supplies
inputs -- an aircraft model class, a mission profile, design parameters -- and reads named
outputs back. It does *not* mean the mission is a pure function: OpenConcept instantiates the
supplied aircraft model class inside every phase, so the boundary is the contract declared in
:mod:`cdadt.mission.contract`.

Convergence is part of the interface
------------------------------------
OpenConcept's mission is a coupled implicit system: phase durations are solved by
``BalanceComp`` against altitude and range targets, throttle is solved for zero horizontal
acceleration, and V1 is solved so the continue and abort distances match. A Newton solver
started at the design mission will not generally converge. :class:`MissionProfile` therefore
carries an explicit continuation schedule -- a sequence of progressively harder mission
profiles -- rather than leaving that as an undocumented sequence of ``set_val`` calls in a run
script. This mirrors what OpenConcept's own ``B738_sizing.py`` example does at lines 472-490,
but as a declared object that can be inspected and tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import openmdao.api as om
from openconcept.mission import FullMissionWithReserve

from cdadt.mission.contract import (
    MISSION_OUTPUTS,
    MISSION_OUTPUTS_BY_NAME,
    MISSION_PHASES,
    MISSION_PROFILE_INPUTS,
    PHASE_SPECS_BY_NAME,
    THROTTLE_BY_PHASE_PATH,
    MissionOutput,
)

__all__ = ["ContinuationStep", "MissionBlackBox", "MissionProfile", "PhaseSchedule"]


@dataclass(frozen=True)
class PhaseSchedule:
    """Airspeed and vertical-speed schedule flown in one steady-flight phase.

    Parameters
    ----------
    equivalent_airspeed : sequence of float
        Equivalent airspeed at each analysis node. Length must equal ``num_nodes``, or be
        2 to be linearly interpolated across the nodes.
    vertical_speed : sequence of float
        Vertical speed at each analysis node, same length convention.
    airspeed_units : str, optional
        Units of ``equivalent_airspeed``. Default ``"kn"``.
    vertical_speed_units : str, optional
        Units of ``vertical_speed``. Default ``"ft/min"``.
    """

    equivalent_airspeed: Sequence[float]
    vertical_speed: Sequence[float]
    airspeed_units: str = "kn"
    vertical_speed_units: str = "ft/min"

    def airspeed_at_nodes(self, num_nodes: int) -> np.ndarray:
        """Return the airspeed schedule resampled to ``num_nodes`` points."""
        return _resample(self.equivalent_airspeed, num_nodes, "equivalent_airspeed")

    def vertical_speed_at_nodes(self, num_nodes: int) -> np.ndarray:
        """Return the vertical-speed schedule resampled to ``num_nodes`` points."""
        return _resample(self.vertical_speed, num_nodes, "vertical_speed")


def _resample(values: Sequence[float], num_nodes: int, label: str) -> np.ndarray:
    """Return ``values`` as a length-``num_nodes`` array, interpolating a two-point schedule.

    Parameters
    ----------
    values : sequence of float
        Either exactly ``num_nodes`` values, or exactly two endpoints to interpolate
        between.
    num_nodes : int
        Target length.
    label : str
        Name of the quantity, used in the error message.

    Returns
    -------
    numpy.ndarray
        Array of length ``num_nodes``.

    Raises
    ------
    ValueError
        If the schedule is neither the right length nor a two-point endpoint pair. Silently
        broadcasting the wrong length would fly a different profile than the one requested.
    """
    array = np.asarray(values, dtype=float)
    if array.size == num_nodes:
        return array
    if array.size == 2:
        return np.linspace(array[0], array[1], num_nodes)
    raise ValueError(
        f"{label} has {array.size} values but the mission has {num_nodes} nodes per phase. "
        f"Supply either {num_nodes} values or 2 endpoints to interpolate between."
    )


@dataclass(frozen=True)
class ContinuationStep:
    """One step of the continuation schedule used to converge the mission.

    Parameters
    ----------
    description : str
        What this step relaxes, for the run log.
    overrides : mapping
        Mission profile values to set for this step, keyed by the names in
        :data:`~cdadt.mission.contract.MISSION_PROFILE_INPUTS`, as
        ``(value, units)`` pairs.
    phase_overrides : mapping, optional
        Per-phase schedule replacements for this step, keyed by phase name.
    """

    description: str
    overrides: Mapping[str, tuple[float, str | None]]
    phase_overrides: Mapping[str, PhaseSchedule] = field(default_factory=dict)


@dataclass(frozen=True)
class MissionProfile:
    """The mission to be flown, and the schedule for converging it.

    Parameters
    ----------
    schedules : mapping
        Airspeed and vertical-speed schedule per steady-flight phase, keyed by phase name.
        Every phase in :data:`~cdadt.mission.contract.MISSION_PHASES` with
        ``kind is PhaseKind.STEADY_FLIGHT`` must be present.
    parameters : mapping
        Values for the mission profile inputs, as ``(value, units)`` pairs. Every name in
        :data:`~cdadt.mission.contract.MISSION_PROFILE_INPUTS` must be present; there are
        no defaults, for the same reason there are none in
        :class:`~cdadt.core.configuration.AircraftConfiguration`.
    continuation : sequence of ContinuationStep, optional
        Progressively harder profiles to run before the real one. Each is applied and the
        model run, so the Newton solver starts each step from a converged neighbour.
        An empty schedule means the mission is attempted directly.
    takeoff_speed_guess : tuple of (float, str), optional
        Initial guess for the true airspeed vectors in the ground-roll phases. These
        integrate acceleration from a standstill and are sensitive to their starting
        point. Default ``(100.0, "kn")``.

    Raises
    ------
    ValueError
        If a required schedule or parameter is absent.
    """

    schedules: Mapping[str, PhaseSchedule]
    parameters: Mapping[str, tuple[float, str | None]]
    continuation: Sequence[ContinuationStep] = ()
    takeoff_speed_guess: tuple[float, str] = (100.0, "kn")
    ground_roll_initial_speed: tuple[float, str] | None = None

    @classmethod
    def from_config(cls, config, root: str = "mission") -> MissionProfile:
        """Build a mission profile from a configuration tree.

        The mission is part of the case, not part of the code, so it is configured
        alongside the airframe rather than written into a run script. The expected layout,
        rooted at ``root``:

        .. code-block:: yaml

           mission:
             mission_range: {value: 2800, units: nmi}      # every MISSION_PROFILE_INPUTS name
             cruise|h0: {value: 35000, units: ft}
             ...
             schedule:
               climb:
                 Ueas: {value: [230, 252], units: kn}      # 2 endpoints or num_nodes values
                 vs:   {value: [2300, 400], units: ft/min}
               ...
             takeoff_speed_guess: {value: 100, units: kn}
             continuation:
               step_0:                                     # applied in sorted key order
                 description_only_in_the_key: ...
                 mission_range: {value: 500, units: nmi}

        Parameters
        ----------
        config : AircraftConfiguration
            Configuration containing the mission definition.
        root : str, optional
            Path prefix the mission lives under. Default ``"mission"``.

        Returns
        -------
        MissionProfile
            The configured mission.

        Raises
        ------
        MissingConfigurationError
            If a required parameter or schedule is absent. There are no defaults.
        """
        prefix = f"{root}|"

        parameters = {}
        for variable in MISSION_PROFILE_INPUTS:
            parameters[variable.name] = (
                config.scalar(f"{prefix}{variable.name}", units=variable.units),
                variable.units,
            )

        schedules = {}
        for spec in MISSION_PHASES:
            if spec.kind.name != "STEADY_FLIGHT":
                continue
            airspeed_path = f"{prefix}schedule|{spec.name}|Ueas"
            vertical_path = f"{prefix}schedule|{spec.name}|vs"
            config.require_all([airspeed_path, vertical_path])
            schedules[spec.name] = PhaseSchedule(
                equivalent_airspeed=config.value(airspeed_path).tolist(),
                vertical_speed=config.value(vertical_path).tolist(),
                airspeed_units=config.units(airspeed_path),
                vertical_speed_units=config.units(vertical_path),
            )

        guess_path = f"{prefix}takeoff_speed_guess"
        initial_speed_path = f"{prefix}ground_roll_initial_speed"
        config.require_all([guess_path, initial_speed_path])
        takeoff_speed_guess = (config.scalar(guess_path), config.units(guess_path))
        ground_roll_initial_speed = (config.scalar(initial_speed_path), config.units(initial_speed_path))

        continuation = []
        continuation_prefix = f"{prefix}continuation|"
        step_names = sorted(
            {n[len(continuation_prefix) :].split("|")[0] for n in config if n.startswith(continuation_prefix)}
        )
        for step_name in step_names:
            step_root = f"{continuation_prefix}{step_name}|"
            schedule_root = f"{step_root}schedule|"
            overrides = {}
            schedule_entries: dict[str, dict[str, str]] = {}
            for name in config:
                if not name.startswith(step_root):
                    continue
                if name.startswith(schedule_root):
                    # continuation|<step>|schedule|<phase>|<Ueas|vs>
                    phase_name, quantity = name[len(schedule_root) :].split("|")
                    schedule_entries.setdefault(phase_name, {})[quantity] = name
                    continue
                overrides[name[len(step_root) :]] = (config.scalar(name), config.units(name))

            phase_overrides = {}
            for phase_name, paths in schedule_entries.items():
                if set(paths) != {"Ueas", "vs"}:
                    raise ValueError(
                        f"Continuation step '{step_name}' overrides the schedule for phase '{phase_name}' "
                        f"but gives {sorted(paths)}; both 'Ueas' and 'vs' are required. Overriding one "
                        f"alone would silently keep the other from the design profile."
                    )
                phase_overrides[phase_name] = PhaseSchedule(
                    equivalent_airspeed=config.value(paths["Ueas"]).tolist(),
                    vertical_speed=config.value(paths["vs"]).tolist(),
                    airspeed_units=config.units(paths["Ueas"]),
                    vertical_speed_units=config.units(paths["vs"]),
                )

            continuation.append(
                ContinuationStep(
                    description=step_name,
                    overrides=overrides,
                    phase_overrides=phase_overrides,
                )
            )

        return cls(
            schedules=schedules,
            parameters=parameters,
            continuation=continuation,
            takeoff_speed_guess=takeoff_speed_guess,
            ground_roll_initial_speed=ground_roll_initial_speed,
        )

    def __post_init__(self) -> None:
        required_phases = {spec.name for spec in MISSION_PHASES if spec.kind.name == "STEADY_FLIGHT"}
        missing_phases = sorted(required_phases - set(self.schedules))
        if missing_phases:
            raise ValueError(
                f"No airspeed/vertical-speed schedule given for {missing_phases}. Every steady-flight "
                f"phase must be scheduled; OpenConcept has no default profile to fall back on."
            )

        missing_parameters = sorted(MISSION_PROFILE_INPUTS.names - set(self.parameters))
        if missing_parameters:
            raise ValueError(f"Mission profile parameters {missing_parameters} were not given and have no defaults.")


class MissionBlackBox:
    """Full mission sizing analysis, consumed as a black box.

    Wraps :class:`openconcept.mission.FullMissionWithReserve`: balanced-field takeoff,
    climb, cruise, descent, the Part 25 reserve mission, and loiter.

    Parameters
    ----------
    aircraft_model_class : type
        The class OpenConcept will instantiate in every phase, normally produced by
        :meth:`~cdadt.mission.aircraft_model.AircraftModelFactory.build`.
    profile : MissionProfile
        The mission to fly and the continuation schedule for converging it.
    num_nodes : int, optional
        Analysis points per phase. Must be odd, because OpenConcept integrates with
        Simpson's rule over ``2N + 1`` points. Default 11.

    Raises
    ------
    ValueError
        If ``num_nodes`` is even.

    Notes
    -----
    Every OpenConcept variable path this class reads or writes is declared in
    :mod:`cdadt.mission.contract`. Nothing here contains a literal path.
    """

    def __init__(self, aircraft_model_class: type, profile: MissionProfile, num_nodes: int = 11) -> None:
        if num_nodes % 2 == 0:
            raise ValueError(
                f"num_nodes must be odd (OpenConcept integrates with Simpson's rule over 2N+1 points); got {num_nodes}"
            )
        self._aircraft_model_class = aircraft_model_class
        self._profile = profile
        self._num_nodes = num_nodes
        self._group_name = "mission"

    @property
    def num_nodes(self) -> int:
        """Return the number of analysis points per phase."""
        return self._num_nodes

    @property
    def profile(self) -> MissionProfile:
        """Return the mission profile being flown."""
        return self._profile

    @property
    def group_name(self) -> str:
        """Return the subsystem name the mission group is added under."""
        return self._group_name

    @property
    def outputs(self) -> tuple[MissionOutput, ...]:
        """Return the results this black box makes available."""
        return MISSION_OUTPUTS

    def build(self, parent: om.Group, name: str = "mission") -> om.Group:
        """Add the mission group to ``parent``.

        Parameters
        ----------
        parent : openmdao.api.Group
            Group to add the mission to. Design parameters under ``ac|`` are promoted from
            the mission up to ``parent``, which is how a single design parameter reaches
            every phase.
        name : str, optional
            Subsystem name for the mission group. Default ``"mission"``.

        Returns
        -------
        openmdao.api.Group
            The mission group that was added.
        """
        self._group_name = name
        return parent.add_subsystem(
            name,
            FullMissionWithReserve(num_nodes=self._num_nodes, aircraft_model=self._aircraft_model_class),
            promotes_inputs=["ac|*"],
        )

    def path(self, output_name: str) -> str:
        """Return the absolute problem path for a named result.

        Parameters
        ----------
        output_name : str
            One of the names in :data:`~cdadt.mission.contract.MISSION_OUTPUTS`.

        Returns
        -------
        str
            Path usable with :meth:`openmdao.api.Problem.get_val`.

        Raises
        ------
        KeyError
            If ``output_name`` is not a declared mission output.
        """
        try:
            output = MISSION_OUTPUTS_BY_NAME[output_name]
        except KeyError:
            raise KeyError(
                f"'{output_name}' is not a declared mission output. Declared: {sorted(MISSION_OUTPUTS_BY_NAME)}"
            ) from None
        return f"{self._group_name}.{output.path}"

    def throttle_path(self, phase_name: str) -> str:
        """Return the absolute problem path for a phase's throttle vector.

        Parameters
        ----------
        phase_name : str
            Name of a mission phase, as it appears in
            :data:`~cdadt.mission.contract.MISSION_PHASES`.

        Returns
        -------
        str
            Path to the throttle vector, used by the thrust-margin certification
            requirement.

        Raises
        ------
        KeyError
            If ``phase_name`` is not a phase OpenConcept builds.
        """
        subsystem = PHASE_SPECS_BY_NAME[phase_name].subsystem
        return f"{self._group_name}.{THROTTLE_BY_PHASE_PATH.format(phase=subsystem)}"

    def read_outputs(self, problem: om.Problem) -> dict[str, np.ndarray]:
        """Read every declared mission output from a run problem.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem containing this mission, already run.

        Returns
        -------
        dict
            Every name in :data:`~cdadt.mission.contract.MISSION_OUTPUTS` mapped to its
            value in the declared units. Reading all of them, rather than the few a caller
            happens to want, is what makes the run report complete by construction.
        """
        return {
            output.name: problem.get_val(f"{self._group_name}.{output.path}", units=output.units)
            for output in MISSION_OUTPUTS
        }

    def set_profile(self, problem: om.Problem) -> None:
        """Set the mission profile on a problem without running it.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem that has been set up and contains this mission.
        """
        self._apply_parameters(problem, self._profile.parameters)
        self._apply_schedules(problem, self._profile.schedules)
        self._apply_takeoff_speed_guesses(problem)
        self._apply_ground_roll_initial_speed(problem)

    def converge(self, problem: om.Problem, verbose: bool = False) -> None:
        """Set the profile and run the continuation schedule up to the design mission.

        Each continuation step is applied on top of the design profile and the model run,
        so the solver enters the next step from a converged neighbouring solution. The
        final ``run_model`` is on the design mission itself.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem that has been set up and contains this mission.
        verbose : bool, optional
            Print each continuation step as it runs. Default ``False``.
        """
        for step in self._profile.continuation:
            if verbose:
                print(f"[cdadt] continuation step: {step.description}")
            self.set_profile(problem)
            self._apply_parameters(problem, step.overrides)
            self._apply_schedules(problem, step.phase_overrides)
            problem.run_model()

        if verbose:
            print("[cdadt] continuation complete; running the design mission")
        self.set_profile(problem)
        problem.run_model()

    # -- internals ----------------------------------------------------------------------

    def _apply_parameters(self, problem: om.Problem, parameters: Mapping[str, tuple[float, str | None]]) -> None:
        """Set mission profile parameters on the problem."""
        for name, (value, units) in parameters.items():
            problem.set_val(f"{self._group_name}.{name}", value, units=units)

    def _apply_schedules(self, problem: om.Problem, schedules: Mapping[str, PhaseSchedule]) -> None:
        """Set the airspeed and vertical-speed vectors for the given phases."""
        for phase_name, schedule in schedules.items():
            problem.set_val(
                f"{self._group_name}.{phase_name}.fltcond|Ueas",
                schedule.airspeed_at_nodes(self._num_nodes),
                units=schedule.airspeed_units,
            )
            problem.set_val(
                f"{self._group_name}.{phase_name}.fltcond|vs",
                schedule.vertical_speed_at_nodes(self._num_nodes),
                units=schedule.vertical_speed_units,
            )

    def _apply_takeoff_speed_guesses(self, problem: om.Problem) -> None:
        """Seed the ground-roll true-airspeed vectors.

        These phases integrate acceleration from a standstill and their solver is sensitive
        to the starting point, so OpenConcept's own example seeds them explicitly. Doing it
        here means a cdadt run does not depend on a caller remembering to.
        """
        value, units = self._profile.takeoff_speed_guess
        for spec in MISSION_PHASES:
            if spec.kind.name == "GROUND_ROLL":
                problem.set_val(
                    f"{self._group_name}.{spec.subsystem}.fltcond|Utrue",
                    np.full(self._num_nodes, value),
                    units=units,
                )

    def _apply_ground_roll_initial_speed(self, problem: om.Problem) -> None:
        """Set the speed the ground roll starts from.

        OpenConcept starts its ground roll at 2 m/s. That is not a physical choice so much
        as a stand-in for zero, and at that speed the skin-friction correlation inside its
        own parasite drag buildup is outside its domain: it evaluates
        ``0.523 / log(0.06 Re)**2`` and ``1.32824 / sqrt(Re)``, both of which return ``NaN``
        as the Reynolds number passes through zero. The failure is real and reproducible --
        it is why OpenConcept's own ``B738_sizing`` example does not converge here -- and
        during an optimization it appears as an "Invalid Number Detected" exit with no
        indication of where it came from.

        cdadt does not modify OpenConcept, so it starts the roll from a configured speed
        instead. This is a modeling decision with a real, if small, consequence: the
        aircraft is credited with the first few metres of acceleration, which shortens the
        computed field length slightly. It is therefore configured with a stated reason
        rather than adjusted quietly until the optimizer stops failing.
        """
        if self._profile.ground_roll_initial_speed is None:
            return
        value, units = self._profile.ground_roll_initial_speed
        for spec in MISSION_PHASES:
            if spec.kind.name == "GROUND_ROLL":
                problem.set_val(f"{self._group_name}.{spec.subsystem}.zero_speed", value, units=units)

    def __repr__(self) -> str:
        """Return a representation naming the model class and node count."""
        return f"MissionBlackBox(aircraft_model={self._aircraft_model_class.__name__}, num_nodes={self._num_nodes})"
