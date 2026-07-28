"""The case file: one YAML document that describes an entire study.

A cdadt run is defined by a file, not by a script. The case file names the black box, the
aircraft, the mission, the solver, and -- for an optimization -- the design variables, the
objective and the certification requirements. Two studies that differ only in which wing
parameters are free differ only in their case files.

That is a design decision with consequences. It means the design variables and the objective
are *data fed into the black box*, so a study can be reproduced, diffed and archived without
reading Python. It also means the file has to be validated hard: a mistyped variable name in a
script fails at the line that mentions it, but a mistyped key in a configuration file fails
much later, somewhere unrelated, or does not fail at all. Every class here therefore rejects
unknown keys, missing keys and wrong types by name, and the black box separately rejects any
variable it does not actually publish.

Layout
------

.. code-block:: yaml

    black_box:
      model: openconcept.examples.B738_sizing:B738SizingMissionAnalysis
      num_nodes: 21
    solver:
      maxiter: 20
      atol: 1.0e-9
      rtol: 1.0e-9
      iprint: -1
      err_on_non_converge: true
    aircraft:
      ac|geom|wing|S_ref: {value: 124.6, units: m**2, source: "b737.org.uk tech specs"}
    initial_guesses:
      ac|weights|MTOW: {value: 50000, units: kg}
    mission:
      parameters:
        mission_range: {value: 2800, units: nmi}
      schedule:
        climb: {Ueas: {value: [230, 252], units: kn}, vs: {value: [2300, 400], units: ft/min}}
      takeoff_speed_guess: {value: 100, units: kn}
      continuation:
        - description: short range at low altitude
          parameters: {mission_range: {value: 500, units: nmi}}
          schedule: {descent: {Ueas: {value: [252, 250], units: kn}, vs: {value: -800, units: ft/min}}}
    optimization:
      driver: {name: SLSQP, maxiter: 50, tol: 1.0e-6, derivative_mode: fwd}
      objective: {name: total_fuel, ref: 2.0e4, sense: minimize}
      design_variables:
        - {name: ac|geom|wing|S_ref, lower: 90, upper: 180, units: m**2}
      requirements:
        - {type: balanced_field_length, limit: 8000, units: ft,
           regulation: "14 CFR 25.113", source: "Design field length, 8000 ft dry, sea level, ISA"}

Every section but ``optimization`` is required. See :doc:`/configuration` for the meaning of
each key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from cdadt.blackbox import SolverSettings
from cdadt.mission import STEADY_FLIGHT_PHASES, ContinuationStep, MissionProfile, PhaseSchedule
from cdadt.parameters import Parameter

__all__ = [
    "BlackBoxConfig",
    "Config",
    "ConfigError",
    "DesignVariableSpec",
    "MissionConfig",
    "ObjectiveSpec",
    "OptimizationConfig",
    "RequirementSpec",
    "SolverConfig",
]


class ConfigError(Exception):
    """Raised when a case file is missing something, has something extra, or has it wrong."""


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or raise naming ``where`` it was expected."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{where}' must be a mapping; got {type(value).__name__}.")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    """Raise if ``data`` has keys outside ``allowed``.

    A silently ignored key is the failure mode a configuration file is most prone to: the run
    succeeds and answers a different question than the one that was asked.
    """
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f"'{where}' has unknown key(s) {unknown}. Allowed keys are {sorted(allowed)}.")


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    """Return ``data[key]``, or raise naming ``where`` it was required."""
    if key not in data:
        raise ConfigError(f"'{where}' requires a '{key}' key.")
    return data[key]


def _quantity(entry: Any, where: str) -> tuple[float, str | None]:
    """Return ``(value, units)`` from a ``{value, units}`` mapping."""
    mapping = _require_mapping(entry, where)
    _reject_unknown(mapping, ("value", "units", "source"), where)
    value = _require(mapping, "value", where)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"'{where}.value' must be a number; got {value!r}.") from error
    return number, mapping.get("units")


class BlackBoxConfig:
    """Which sizing analysis to drive, and on what grid.

    Parameters
    ----------
    model : str
        The analysis to load, as ``"module.path:ClassName"``.
    num_nodes : int
        Analysis points per mission phase. Must be odd.
    """

    __slots__ = ("_model", "_num_nodes")

    ALLOWED = ("model", "num_nodes")

    def __init__(self, model: str, num_nodes: int) -> None:
        self._model = str(model)
        self._num_nodes = int(num_nodes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BlackBoxConfig:
        """Build from the ``black_box`` section."""
        _reject_unknown(data, cls.ALLOWED, "black_box")
        return cls(model=_require(data, "model", "black_box"), num_nodes=_require(data, "num_nodes", "black_box"))

    @property
    def model(self) -> str:
        """The ``module:Class`` string naming the analysis."""
        return self._model

    @property
    def num_nodes(self) -> int:
        """Analysis points per mission phase."""
        return self._num_nodes

    def __repr__(self) -> str:
        """Return a representation naming the model and grid."""
        return f"BlackBoxConfig({self._model!r}, num_nodes={self._num_nodes})"


class SolverConfig:
    """How hard to converge the black box.

    Parameters
    ----------
    maxiter : int, optional
        Newton iteration limit. Default 20.
    atol, rtol : float, optional
        Residual tolerances. Default 1e-9.
    iprint : int, optional
        Solver print level. Default ``-1``.
    err_on_non_converge : bool, optional
        Whether a failed solve raises. Default ``True``.
    """

    __slots__ = ("_atol", "_err_on_non_converge", "_iprint", "_maxiter", "_rtol")

    ALLOWED = ("maxiter", "atol", "rtol", "iprint", "err_on_non_converge")

    def __init__(
        self,
        maxiter: int = 20,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        iprint: int = -1,
        err_on_non_converge: bool = True,
    ) -> None:
        self._maxiter = int(maxiter)
        self._atol = float(atol)
        self._rtol = float(rtol)
        self._iprint = int(iprint)
        self._err_on_non_converge = bool(err_on_non_converge)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolverConfig:
        """Build from the ``solver`` section, which may be empty."""
        _reject_unknown(data, cls.ALLOWED, "solver")
        return cls(**{key: data[key] for key in cls.ALLOWED if key in data})

    def settings(self) -> SolverSettings:
        """Return the :class:`~cdadt.blackbox.SolverSettings` this section describes."""
        return SolverSettings(
            maxiter=self._maxiter,
            atol=self._atol,
            rtol=self._rtol,
            iprint=self._iprint,
            err_on_non_converge=self._err_on_non_converge,
        )

    def __repr__(self) -> str:
        """Return a representation naming the iteration limit."""
        return f"SolverConfig(maxiter={self._maxiter})"


class MissionConfig:
    """The mission section: what is flown, and the ladder that converges it.

    Parameters
    ----------
    parameters : mapping
        Mission-level values as ``name -> (value, units)``.
    schedules : mapping
        One :class:`~cdadt.mission.PhaseSchedule` per steady-flight phase.
    continuation : sequence of ContinuationStep, optional
        The ladder of easier missions.
    takeoff_speed_guess : tuple, optional
        Ground-roll true-airspeed seed as ``(value, units)``. Default ``(100.0, "kn")``.
    mission_path : str, optional
        Name of the mission subsystem inside the black box. Default ``"mission"``.
    """

    __slots__ = ("_continuation", "_mission_path", "_parameters", "_schedules", "_takeoff_speed_guess")

    ALLOWED = ("parameters", "schedule", "continuation", "takeoff_speed_guess", "mission_path")
    STEP_ALLOWED = ("description", "parameters", "schedule")

    def __init__(
        self,
        parameters: Mapping[str, tuple[float, str | None]],
        schedules: Mapping[str, PhaseSchedule],
        continuation: Sequence[ContinuationStep] = (),
        takeoff_speed_guess: tuple[float, str] = (100.0, "kn"),
        mission_path: str = "mission",
    ) -> None:
        self._parameters = dict(parameters)
        self._schedules = dict(schedules)
        self._continuation = tuple(continuation)
        self._takeoff_speed_guess = takeoff_speed_guess
        self._mission_path = str(mission_path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MissionConfig:
        """Build from the ``mission`` section."""
        _reject_unknown(data, cls.ALLOWED, "mission")
        guess_value, guess_units = _quantity(
            data.get("takeoff_speed_guess", {"value": 100.0, "units": "kn"}), "mission.takeoff_speed_guess"
        )
        steps = []
        for index, entry in enumerate(data.get("continuation", [])):
            where = f"mission.continuation[{index}]"
            step = _require_mapping(entry, where)
            _reject_unknown(step, cls.STEP_ALLOWED, where)
            steps.append(
                ContinuationStep(
                    description=str(step.get("description", f"step {index + 1}")),
                    parameters=cls._read_parameters(step.get("parameters", {}), f"{where}.parameters"),
                    schedules=cls._read_schedules(step.get("schedule", {}), f"{where}.schedule", complete=False),
                )
            )
        return cls(
            parameters=cls._read_parameters(_require(data, "parameters", "mission"), "mission.parameters"),
            schedules=cls._read_schedules(_require(data, "schedule", "mission"), "mission.schedule", complete=True),
            continuation=steps,
            takeoff_speed_guess=(guess_value, guess_units or "kn"),
            mission_path=str(data.get("mission_path", "mission")),
        )

    @staticmethod
    def _read_parameters(entries: Any, where: str) -> dict[str, tuple[float, str | None]]:
        """Return ``name -> (value, units)`` from a mapping of quantities."""
        mapping = _require_mapping(entries, where)
        return {name: _quantity(entry, f"{where}.{name}") for name, entry in mapping.items()}

    @staticmethod
    def _read_schedules(entries: Any, where: str, complete: bool) -> dict[str, PhaseSchedule]:
        """Return ``phase -> PhaseSchedule``.

        Parameters
        ----------
        entries : mapping
            Phase entries, each giving both ``Ueas`` and ``vs``.
        where : str
            Path in the case file, for error messages.
        complete : bool
            Whether every steady-flight phase must appear. ``True`` for the design mission,
            ``False`` for a continuation step, which overrides only what it names.

        Raises
        ------
        ConfigError
            If a phase gives one of ``Ueas``/``vs`` without the other -- overriding one alone
            silently keeps the other from whatever was set before, which inside a continuation
            step means flying a profile nobody wrote down -- or if a required phase is missing,
            or if a phase is not one the box flies.
        """
        mapping = _require_mapping(entries, where)
        unknown = sorted(set(mapping) - set(STEADY_FLIGHT_PHASES))
        if unknown:
            raise ConfigError(
                f"'{where}' schedules phase(s) {unknown}, which the mission does not fly. "
                f"The steady-flight phases are {list(STEADY_FLIGHT_PHASES)}."
            )
        if complete:
            missing = [phase for phase in STEADY_FLIGHT_PHASES if phase not in mapping]
            if missing:
                raise ConfigError(f"'{where}' is missing a schedule for {missing}.")

        schedules: dict[str, PhaseSchedule] = {}
        for phase, entry in mapping.items():
            phase_where = f"{where}.{phase}"
            phase_entry = _require_mapping(entry, phase_where)
            if set(phase_entry) != {"Ueas", "vs"}:
                raise ConfigError(
                    f"'{phase_where}' gives {sorted(phase_entry)}; both 'Ueas' and 'vs' are required, "
                    f"because setting one alone flies a profile that is half inherited."
                )
            airspeed = _require_mapping(phase_entry["Ueas"], f"{phase_where}.Ueas")
            vertical = _require_mapping(phase_entry["vs"], f"{phase_where}.vs")
            schedules[phase] = PhaseSchedule(
                equivalent_airspeed=_require(airspeed, "value", f"{phase_where}.Ueas"),
                vertical_speed=_require(vertical, "value", f"{phase_where}.vs"),
                airspeed_units=airspeed.get("units", "kn"),
                vertical_speed_units=vertical.get("units", "ft/min"),
            )
        return schedules

    def profile(self) -> MissionProfile:
        """Return the :class:`~cdadt.mission.MissionProfile` this section describes."""
        return MissionProfile(
            parameters=self._parameters,
            schedules=self._schedules,
            continuation=self._continuation,
            takeoff_speed_guess=self._takeoff_speed_guess,
            mission_path=self._mission_path,
        )

    @property
    def mission_path(self) -> str:
        """Name of the mission subsystem inside the black box."""
        return self._mission_path

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Mission-level parameter names, as the box publishes them (with the mission prefix)."""
        return tuple(f"{self._mission_path}.{name}" for name in self._parameters)

    def __repr__(self) -> str:
        """Return a representation naming the phase and step counts."""
        return f"MissionConfig({len(self._schedules)} phases, {len(self._continuation)} continuation steps)"


class DesignVariableSpec:
    """One quantity the optimizer may change.

    Parameters
    ----------
    name : str
        Name of a variable the black box lets a caller set. Anything the box computes is not
        free to choose, and is rejected before the driver runs.
    lower, upper : float
        Bounds, in ``units``.
    units : str or None, optional
        Units of the bounds.
    ref : float or None, optional
        Value that scales the variable to order one for the optimizer. Defaults to the larger
        of ``|lower|`` and ``|upper|``, which is nearly always the right order of magnitude and
        is the difference between an optimizer that converges and one that stalls when a wing
        area in square metres shares a design space with an aspect ratio.
    """

    __slots__ = ("_lower", "_name", "_ref", "_units", "_upper")

    ALLOWED = ("name", "lower", "upper", "units", "ref")

    def __init__(
        self, name: str, lower: float, upper: float, units: str | None = None, ref: float | None = None
    ) -> None:
        self._name = str(name)
        self._lower = float(lower)
        self._upper = float(upper)
        if self._lower >= self._upper:
            raise ConfigError(f"Design variable '{self._name}' has lower {self._lower} >= upper {self._upper}.")
        self._units = units
        self._ref = None if ref is None else float(ref)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> DesignVariableSpec:
        """Build one design variable entry."""
        _reject_unknown(data, cls.ALLOWED, where)
        return cls(
            name=_require(data, "name", where),
            lower=_require(data, "lower", where),
            upper=_require(data, "upper", where),
            units=data.get("units"),
            ref=data.get("ref"),
        )

    @property
    def name(self) -> str:
        """Name of the black-box variable."""
        return self._name

    @property
    def lower(self) -> float:
        """Lower bound, in :attr:`units`."""
        return self._lower

    @property
    def upper(self) -> float:
        """Upper bound, in :attr:`units`."""
        return self._upper

    @property
    def units(self) -> str | None:
        """Units of the bounds."""
        return self._units

    @property
    def ref(self) -> float:
        """Scaling reference, defaulted from the bounds if not given."""
        if self._ref is not None:
            return self._ref
        return max(abs(self._lower), abs(self._upper)) or 1.0

    def __repr__(self) -> str:
        """Return a representation naming the variable and its bounds."""
        return f"DesignVariableSpec({self._name!r}, {self._lower}, {self._upper}, {self._units!r})"


class ObjectiveSpec:
    """What the optimizer minimizes or maximizes.

    Parameters
    ----------
    name : str
        Either the name of a discipline response -- ``"total_fuel"``, ``"MTOW"`` -- or a raw
        black-box path. Response names are preferred: they carry their own units.
    units : str or None, optional
        Units to optimize in. ``None`` takes the response's own units, or the box's for a raw
        path.
    ref : float or None, optional
        Value that scales the objective to order one. It matters: SciPy's SLSQP takes its
        finite-difference step and its convergence test on the *scaled* objective, so an
        objective of order 1e4 is effectively converged before it starts.
    sense : str, optional
        ``"minimize"`` or ``"maximize"``. Default ``"minimize"``.
    """

    __slots__ = ("_name", "_ref", "_sense", "_units")

    ALLOWED = ("name", "units", "ref", "sense")
    SENSES = ("minimize", "maximize")

    def __init__(self, name: str, units: str | None = None, ref: float | None = None, sense: str = "minimize") -> None:
        if sense not in self.SENSES:
            raise ConfigError(f"Objective sense must be one of {list(self.SENSES)}; got {sense!r}.")
        self._name = str(name)
        self._units = units
        self._ref = None if ref is None else float(ref)
        self._sense = sense

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObjectiveSpec:
        """Build from the ``optimization.objective`` section."""
        where = "optimization.objective"
        _reject_unknown(data, cls.ALLOWED, where)
        return cls(
            name=_require(data, "name", where),
            units=data.get("units"),
            ref=data.get("ref"),
            sense=str(data.get("sense", "minimize")),
        )

    @property
    def name(self) -> str:
        """Response name or black-box path being optimized."""
        return self._name

    @property
    def units(self) -> str | None:
        """Units the objective is optimized in, if the case file fixed them."""
        return self._units

    @property
    def ref(self) -> float | None:
        """Objective scaling, or ``None`` for unscaled."""
        return self._ref

    @property
    def sense(self) -> str:
        """``"minimize"`` or ``"maximize"``."""
        return self._sense

    @property
    def scaler(self) -> float:
        """Return ``+1`` to minimize or ``-1`` to maximize.

        OpenMDAO drivers always minimize, so a maximization is a minimization of the negated
        objective. Doing it here keeps the sign in one place.
        """
        return 1.0 if self._sense == "minimize" else -1.0

    def __repr__(self) -> str:
        """Return a representation naming the objective and its sense."""
        return f"ObjectiveSpec({self._name!r}, sense={self._sense!r})"


class RequirementSpec:
    """One certification requirement, as the case file states it.

    Parameters
    ----------
    kind : str
        Which requirement class to build, e.g. ``"balanced_field_length"``. See
        :mod:`cdadt.certification` for what is available.
    limit : float
        The limiting value, in ``units``.
    regulation : str
        The regulation the limit comes from, e.g. ``"14 CFR 25.113"``. Required: a constraint
        in a certification-driven study that cannot name its regulation is a constraint nobody
        can defend.
    source : str
        Where the *number* comes from -- which runway, which aerodrome category, which
        aeroplane class. Required, for the same reason.
    units : str or None, optional
        Units of ``limit``.
    options : mapping, optional
        Extra keys the requirement class needs, such as which phase a throttle limit applies
        to. Validated by that class, not here.
    """

    __slots__ = ("_kind", "_limit", "_options", "_regulation", "_source", "_units")

    RESERVED = ("type", "limit", "units", "regulation", "source")

    def __init__(
        self,
        kind: str,
        limit: float,
        regulation: str,
        source: str,
        units: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._kind = str(kind)
        self._limit = float(limit)
        self._regulation = str(regulation)
        self._source = str(source)
        self._units = units
        self._options = dict(options or {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> RequirementSpec:
        """Build one requirement entry, keeping unreserved keys as class options."""
        mapping = _require_mapping(data, where)
        return cls(
            kind=_require(mapping, "type", where),
            limit=_require(mapping, "limit", where),
            regulation=_require(mapping, "regulation", where),
            source=_require(mapping, "source", where),
            units=mapping.get("units"),
            options={key: value for key, value in mapping.items() if key not in cls.RESERVED},
        )

    @property
    def kind(self) -> str:
        """Which requirement class to build."""
        return self._kind

    @property
    def limit(self) -> float:
        """The limiting value."""
        return self._limit

    @property
    def units(self) -> str | None:
        """Units of the limit."""
        return self._units

    @property
    def regulation(self) -> str:
        """The regulation the limit comes from."""
        return self._regulation

    @property
    def source(self) -> str:
        """Where the number comes from."""
        return self._source

    @property
    def options(self) -> Mapping[str, Any]:
        """Extra keys for the requirement class."""
        return dict(self._options)

    def __repr__(self) -> str:
        """Return a representation naming the requirement and its limit."""
        return f"RequirementSpec({self._kind!r}, limit={self._limit}, regulation={self._regulation!r})"


class OptimizationConfig:
    """The optimization section: driver, objective, design variables, requirements.

    Parameters
    ----------
    objective : ObjectiveSpec
        What to optimize.
    design_variables : sequence of DesignVariableSpec
        What may change. At least one is required.
    requirements : sequence of RequirementSpec, optional
        The certification basis. May be empty: an unconstrained optimization is a valid study,
        and the report says so rather than printing an empty table.
    driver : str, optional
        ``"SLSQP"`` (SciPy, always available) or a pyOptSparse optimizer such as ``"IPOPT"`` or
        ``"SNOPT"``. Default ``"SLSQP"``.
    maxiter : int, optional
        Optimizer iteration limit. Default 50.
    tol : float, optional
        Optimizer convergence tolerance. Default 1e-6.
    derivative_mode : str, optional
        ``"auto"``, ``"fwd"`` or ``"rev"``. Default ``"fwd"``: a sizing optimization has a
        handful of design variables and many vector responses, which is forward's case.
    driver_options : mapping, optional
        Extra optimizer settings passed through to pyOptSparse.
    """

    __slots__ = (
        "_derivative_mode",
        "_design_variables",
        "_driver",
        "_driver_options",
        "_maxiter",
        "_objective",
        "_requirements",
        "_tol",
    )

    ALLOWED = ("driver", "objective", "design_variables", "requirements")
    DRIVER_ALLOWED = ("name", "maxiter", "tol", "derivative_mode", "options")

    def __init__(
        self,
        objective: ObjectiveSpec,
        design_variables: Sequence[DesignVariableSpec],
        requirements: Sequence[RequirementSpec] = (),
        driver: str = "SLSQP",
        maxiter: int = 50,
        tol: float = 1e-6,
        derivative_mode: str = "fwd",
        driver_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not design_variables:
            raise ConfigError("An optimization needs at least one design variable.")
        duplicates = sorted({d.name for d in design_variables if [x.name for x in design_variables].count(d.name) > 1})
        if duplicates:
            raise ConfigError(f"Design variable(s) {duplicates} are declared more than once.")
        if derivative_mode not in ("auto", "fwd", "rev"):
            raise ConfigError(f"derivative_mode must be 'auto', 'fwd' or 'rev'; got {derivative_mode!r}.")
        self._objective = objective
        self._design_variables = tuple(design_variables)
        self._requirements = tuple(requirements)
        self._driver = str(driver)
        self._maxiter = int(maxiter)
        self._tol = float(tol)
        self._derivative_mode = derivative_mode
        self._driver_options = dict(driver_options or {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationConfig:
        """Build from the ``optimization`` section."""
        _reject_unknown(data, cls.ALLOWED, "optimization")
        driver = _require_mapping(data.get("driver", {}), "optimization.driver")
        _reject_unknown(driver, cls.DRIVER_ALLOWED, "optimization.driver")

        variables = _require(data, "design_variables", "optimization")
        if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
            raise ConfigError("'optimization.design_variables' must be a list.")

        requirements = data.get("requirements", [])
        if not isinstance(requirements, Sequence) or isinstance(requirements, (str, bytes)):
            raise ConfigError("'optimization.requirements' must be a list.")

        return cls(
            objective=ObjectiveSpec.from_dict(
                _require_mapping(_require(data, "objective", "optimization"), "optimization.objective")
            ),
            design_variables=[
                DesignVariableSpec.from_dict(
                    _require_mapping(entry, f"optimization.design_variables[{index}]"),
                    f"optimization.design_variables[{index}]",
                )
                for index, entry in enumerate(variables)
            ],
            requirements=[
                RequirementSpec.from_dict(entry, f"optimization.requirements[{index}]")
                for index, entry in enumerate(requirements)
            ],
            driver=str(driver.get("name", "SLSQP")),
            maxiter=int(driver.get("maxiter", 50)),
            tol=float(driver.get("tol", 1e-6)),
            derivative_mode=str(driver.get("derivative_mode", "fwd")),
            driver_options=driver.get("options", {}),
        )

    @property
    def objective(self) -> ObjectiveSpec:
        """What to optimize."""
        return self._objective

    @property
    def design_variables(self) -> tuple[DesignVariableSpec, ...]:
        """What may change."""
        return self._design_variables

    @property
    def requirements(self) -> tuple[RequirementSpec, ...]:
        """The certification basis, as stated in the case file."""
        return self._requirements

    @property
    def driver(self) -> str:
        """Optimizer name."""
        return self._driver

    @property
    def maxiter(self) -> int:
        """Optimizer iteration limit."""
        return self._maxiter

    @property
    def tol(self) -> float:
        """Optimizer convergence tolerance."""
        return self._tol

    @property
    def derivative_mode(self) -> str:
        """Total-derivative mode."""
        return self._derivative_mode

    @property
    def driver_options(self) -> Mapping[str, Any]:
        """Extra optimizer settings."""
        return dict(self._driver_options)

    def __repr__(self) -> str:
        """Return a representation naming the objective and the counts."""
        return (
            f"OptimizationConfig(objective={self._objective.name!r}, "
            f"design_variables={len(self._design_variables)}, requirements={len(self._requirements)})"
        )


class Config:
    """A whole case file.

    Parameters
    ----------
    black_box : BlackBoxConfig
        Which sizing analysis to drive.
    aircraft : sequence of Parameter
        The design parameters to set on it.
    mission : MissionConfig
        What is flown.
    solver : SolverConfig, optional
        How hard to converge. Defaults to OpenConcept's own sizing settings.
    initial_guesses : sequence of Parameter, optional
        Starting values for the box's coupled states. Not design parameters: they decide
        whether the solver converges, not what it converges to.
    optimization : OptimizationConfig or None, optional
        Present for an optimization study, absent for a sizing run.
    path : pathlib.Path or None, optional
        Where the file was read from, for messages.

    Examples
    --------
    >>> config = Config.from_yaml("cases/b738.yaml")
    >>> config.black_box.num_nodes
    21
    """

    __slots__ = ("_aircraft", "_black_box", "_initial_guesses", "_mission", "_optimization", "_path", "_solver")

    ALLOWED = ("black_box", "solver", "aircraft", "initial_guesses", "mission", "optimization")

    def __init__(
        self,
        black_box: BlackBoxConfig,
        aircraft: Sequence[Parameter],
        mission: MissionConfig,
        solver: SolverConfig | None = None,
        initial_guesses: Sequence[Parameter] = (),
        optimization: OptimizationConfig | None = None,
        path: Path | None = None,
    ) -> None:
        self._black_box = black_box
        self._aircraft = tuple(aircraft)
        self._mission = mission
        self._solver = solver if solver is not None else SolverConfig()
        self._initial_guesses = tuple(initial_guesses)
        self._optimization = optimization
        self._path = path

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a case file.

        Parameters
        ----------
        path : str or pathlib.Path
            File to read.

        Raises
        ------
        ConfigError
            If the file is not valid YAML, or is not a mapping.
        """
        location = Path(path)
        try:
            with open(location, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ConfigError(f"'{location}' is not valid YAML: {error}") from error
        if not isinstance(data, Mapping):
            raise ConfigError(f"'{location}' must contain a mapping at the top level.")
        return cls.from_dict(data, path=location)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: Path | None = None) -> Config:
        """Build a configuration from an already-parsed mapping."""
        _reject_unknown(data, cls.ALLOWED, "the case file")
        return cls(
            black_box=BlackBoxConfig.from_dict(
                _require_mapping(_require(data, "black_box", "the case file"), "black_box")
            ),
            aircraft=cls._read_parameters(_require(data, "aircraft", "the case file"), "aircraft"),
            mission=MissionConfig.from_dict(_require_mapping(_require(data, "mission", "the case file"), "mission")),
            solver=SolverConfig.from_dict(_require_mapping(data.get("solver", {}), "solver")),
            initial_guesses=cls._read_parameters(data.get("initial_guesses", {}), "initial_guesses"),
            optimization=(
                OptimizationConfig.from_dict(_require_mapping(data["optimization"], "optimization"))
                if "optimization" in data
                else None
            ),
            path=path,
        )

    @staticmethod
    def _read_parameters(entries: Any, where: str) -> list[Parameter]:
        """Return a list of :class:`~cdadt.parameters.Parameter` from a mapping of quantities."""
        mapping = _require_mapping(entries, where)
        parameters = []
        for name, entry in mapping.items():
            location = f"{where}.{name}"
            quantity = _require_mapping(entry, location)
            _reject_unknown(quantity, ("value", "units", "source"), location)
            try:
                parameters.append(
                    Parameter(
                        name=name,
                        value=_require(quantity, "value", location),
                        units=quantity.get("units"),
                        source=str(quantity.get("source", "")),
                    )
                )
            except ValueError as error:
                raise ConfigError(f"'{location}': {error}") from error
        return parameters

    # -- state ---------------------------------------------------------------------------

    @property
    def black_box(self) -> BlackBoxConfig:
        """Which sizing analysis to drive."""
        return self._black_box

    @property
    def aircraft(self) -> tuple[Parameter, ...]:
        """The design parameters set on the box."""
        return self._aircraft

    @property
    def mission(self) -> MissionConfig:
        """What is flown."""
        return self._mission

    @property
    def solver(self) -> SolverConfig:
        """How hard to converge the box."""
        return self._solver

    @property
    def initial_guesses(self) -> tuple[Parameter, ...]:
        """Starting values for the box's coupled states."""
        return self._initial_guesses

    @property
    def optimization(self) -> OptimizationConfig | None:
        """The optimization section, or ``None`` for a sizing-only case."""
        return self._optimization

    @property
    def path(self) -> Path | None:
        """Where the case file was read from, if it was read from a file."""
        return self._path

    @property
    def is_optimization(self) -> bool:
        """Whether this case declares an optimization."""
        return self._optimization is not None

    def __repr__(self) -> str:
        """Return a representation naming the file and what it declares."""
        kind = "optimization" if self.is_optimization else "sizing"
        return f"Config({str(self._path) if self._path else 'in memory'}, {kind}, {len(self._aircraft)} parameters)"
