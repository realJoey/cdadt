"""The case file: one YAML document that describes an entire study.

The layout follows OpenConcept's own B738 run scripts, block for block, so that anyone who can
read ``B738.py`` can read a cdadt case.

``design_variables``
    ``B738.py`` line 86 reads *"Define a bunch of design variables and airplane-specific
    parameters"* and then lists every ``ac|`` variable through
    ``dv_comp.add_output_from_dict(...)``. That block is this one: every design variable, with
    its value, its units and where the number came from. A variable becomes free for the
    optimizer by gaining an ``optimize:`` entry with bounds -- nothing else moves between
    sections, and a study that frees one more variable differs by three lines.

``initial_conditions``
    ``set_values(prob, num_nodes)`` in ``B738.py``, and the first half of
    ``set_mission_profile(prob)`` in ``B738_sizing.py``: the per-phase speed and vertical-speed
    schedules, the cruise altitude, the range, and the solver's starting guesses. Written with
    the same names those functions use.

``continuation``
    The second half of ``set_mission_profile``: the easy mission converged first, then stepped
    up. OpenConcept writes it as two ``run_model()`` calls between blocks of assignments; here
    it is a list of rungs.

``driver``, ``constraints``, ``objective``
    What the examples that *do* optimize -- ``B738_aerostructural.py``,
    ``B738_VLM_drag.py`` -- write as ``add_constraint``/``add_objective`` calls. Every form
    those calls accept is expressible: one-sided, two-sided, equality, with any of OpenMDAO's
    scaling and index arguments.

Every section rejects keys it does not recognise. A silently ignored key in a configuration
file means the run succeeds and answers a different question than the one that was asked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cdadt.blackbox import SolverSettings
from cdadt.mission import ContinuationLadder, ContinuationStep, InitialConditions
from cdadt.parameters import Parameter

__all__ = [
    "BlackBoxConfig",
    "Bounds",
    "Config",
    "ConfigError",
    "ConstraintSpec",
    "DriverConfig",
    "ObjectiveSpec",
    "OptimizeSpec",
    "Scaling",
    "SolverConfig",
    "VariableSpec",
]


class ConfigError(Exception):
    """Raised when a case file is missing something, has something extra, or has it wrong."""


# =============================================================================================
# Readers
# =============================================================================================


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or raise naming ``where`` it was expected."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{where}' must be a mapping; got {type(value).__name__}.")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    """Return ``value`` as a list, or raise naming ``where`` it was expected."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"'{where}' must be a list; got {type(value).__name__}.")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    """Raise if ``data`` has keys outside ``allowed``."""
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f"'{where}' has unknown key(s) {unknown}. Allowed keys are {sorted(allowed)}.")


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    """Return ``data[key]``, or raise naming ``where`` it was required."""
    if key not in data:
        raise ConfigError(f"'{where}' requires a '{key}' key.")
    return data[key]


def _number(value: Any, where: str) -> float:
    """Return ``value`` as a float, or raise naming ``where`` it came from."""
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"'{where}' must be a number; got {value!r}.") from error


def _number_or_vector(value: Any, where: str) -> Any:
    """Return a float, or an array when the case file gave a sequence.

    A sequence is how a per-node schedule is written -- ``[2300, 400]`` for the endpoints of a
    climb rate -- and how a vector design variable is bounded element by element, which is what
    OpenConcept's aerostructural example does for a spanwise thickness distribution.
    """
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray([float(item) for item in value])
        except (TypeError, ValueError) as error:
            raise ConfigError(f"'{where}' has a non-numeric entry: {value!r}.") from error
    return _number(value, where)


# =============================================================================================
# Value objects shared by design variables, constraints and the objective
# =============================================================================================


class Scaling:
    """How a quantity is scaled for the driver: ``ref``/``ref0`` or ``scaler``/``adder``.

    OpenMDAO accepts either pair and refuses both, because they are two spellings of the same
    affine map. cdadt refuses both here instead, where the offending key can be named.
    """

    __slots__ = ("_adder", "_ref", "_ref0", "_scaler")

    ALLOWED = ("ref", "ref0", "scaler", "adder")

    def __init__(
        self,
        ref: float | None = None,
        ref0: float | None = None,
        scaler: float | None = None,
        adder: float | None = None,
    ) -> None:
        if (ref is not None or ref0 is not None) and (scaler is not None or adder is not None):
            raise ConfigError(
                "Give either 'ref'/'ref0' or 'scaler'/'adder', not both: they are two spellings "
                "of the same scaling, and OpenMDAO refuses the pair."
            )
        self._ref = None if ref is None else float(ref)
        self._ref0 = None if ref0 is None else float(ref0)
        self._scaler = None if scaler is None else float(scaler)
        self._adder = None if adder is None else float(adder)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scaling:
        """Build from whichever of the four keys an entry carries."""
        return cls(**{key: data[key] for key in cls.ALLOWED if key in data})

    @property
    def ref(self) -> float | None:
        """Value the driver should see as one, if given."""
        return self._ref

    @property
    def ref0(self) -> float | None:
        """Value the driver should see as zero, if given."""
        return self._ref0

    @property
    def scaler(self) -> float | None:
        """Multiplicative scaling, if given."""
        return self._scaler

    @property
    def adder(self) -> float | None:
        """Additive offset, if given."""
        return self._adder

    @property
    def is_empty(self) -> bool:
        """Whether no scaling at all was given."""
        return all(value is None for value in (self._ref, self._ref0, self._scaler, self._adder))

    def as_kwargs(self, default_ref: float | None = None) -> dict[str, float]:
        """Return the keyword arguments OpenMDAO takes, omitting anything not given."""
        if self.is_empty and default_ref is not None:
            return {"ref": float(default_ref)}
        return {
            key: value
            for key, value in (
                ("ref", self._ref),
                ("ref0", self._ref0),
                ("scaler", self._scaler),
                ("adder", self._adder),
            )
            if value is not None
        }

    def __repr__(self) -> str:
        """Return a representation naming whichever scaling was given."""
        return f"Scaling({self.as_kwargs()})"


class Bounds:
    """The bound of a constraint: an upper limit, a lower limit, both, or an equality.

    Every form ``add_constraint`` accepts, because every one appears in OpenConcept's own
    examples: a one-sided limit, a two-sided band such as ``lower: 0.01, upper: 1.05`` on a
    throttle history, and an equality such as ``equals: 0``.
    """

    __slots__ = ("_equals", "_lower", "_upper")

    def __init__(self, lower: Any = None, upper: Any = None, equals: Any = None) -> None:
        if equals is not None and (lower is not None or upper is not None):
            raise ConfigError("A constraint is either an equality or an inequality, not both.")
        if equals is None and lower is None and upper is None:
            raise ConfigError("A constraint needs a 'lower', an 'upper' or an 'equals'.")
        self._lower = None if lower is None else _number_or_vector(lower, "constraint lower bound")
        self._upper = None if upper is None else _number_or_vector(upper, "constraint upper bound")
        self._equals = None if equals is None else _number_or_vector(equals, "constraint equality")
        if (
            self._lower is not None
            and self._upper is not None
            and np.any(np.asarray(self._lower) >= np.asarray(self._upper))
        ):
            raise ConfigError(f"Constraint band is empty: lower {lower} >= upper {upper}.")

    @property
    def lower(self) -> Any:
        """The lower side, if there is one."""
        return self._lower

    @property
    def upper(self) -> Any:
        """The upper side, if there is one."""
        return self._upper

    @property
    def equals(self) -> Any:
        """The equality value, if this is an equality."""
        return self._equals

    @property
    def is_equality(self) -> bool:
        """Whether this is an equality constraint."""
        return self._equals is not None

    @property
    def is_two_sided(self) -> bool:
        """Whether both sides are bounded."""
        return self._lower is not None and self._upper is not None

    @property
    def magnitude(self) -> float:
        """A representative size of the bound, used to scale the constraint by default."""
        sides = [side for side in (self._lower, self._upper, self._equals) if side is not None]
        return float(max(np.max(np.abs(side)) for side in sides)) or 1.0

    def as_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments OpenMDAO's ``add_constraint`` takes."""
        if self.is_equality:
            return {"equals": self._equals}
        return {key: value for key, value in (("lower", self._lower), ("upper", self._upper)) if value is not None}

    def describe(self) -> str:
        """Return the bound as it should read in a report."""

        def show(value: Any) -> str:
            array = np.atleast_1d(np.asarray(value))
            return f"{float(array.ravel()[0]):g}" if array.size == 1 else f"{array.size} values"

        if self.is_equality:
            return f"= {show(self._equals)}"
        if self.is_two_sided:
            return f"{show(self._lower)} to {show(self._upper)}"
        if self._upper is not None:
            return f"<= {show(self._upper)}"
        return f">= {show(self._lower)}"

    def __repr__(self) -> str:
        """Return a representation naming the bound."""
        return f"Bounds({self.describe()})"


class OptimizeSpec:
    """The ``optimize:`` entry that frees a design variable for the driver.

    Parameters
    ----------
    bounds : Bounds
        The interval the variable may move in.
    scaling : Scaling, optional
        Driver scaling. Defaults to ``ref`` at the larger bound magnitude, which is nearly
        always the right order and is the difference between an optimizer that converges and one
        that stalls when a wing area in square metres shares a design space with an aspect
        ratio.
    indices : sequence of int or None, optional
        Which elements of a vector variable are free. ``None`` frees all of them.
    """

    __slots__ = ("_bounds", "_indices", "_scaling")

    ALLOWED = ("lower", "upper", "indices", *Scaling.ALLOWED)

    def __init__(self, bounds: Bounds, scaling: Scaling | None = None, indices: Sequence[int] | None = None) -> None:
        if bounds.is_equality:
            raise ConfigError("A design variable is bounded by 'lower' and 'upper', not by 'equals'.")
        self._bounds = bounds
        self._scaling = scaling if scaling is not None else Scaling()
        self._indices = None if indices is None else [int(index) for index in indices]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> OptimizeSpec:
        """Build one ``optimize:`` entry."""
        mapping = _mapping(data, where)
        _reject_unknown(mapping, cls.ALLOWED, where)
        try:
            return cls(
                bounds=Bounds(lower=mapping.get("lower"), upper=mapping.get("upper")),
                scaling=Scaling.from_dict(mapping),
                indices=mapping.get("indices"),
            )
        except ConfigError as error:
            raise ConfigError(f"'{where}': {error}") from None

    @property
    def bounds(self) -> Bounds:
        """The interval the variable may move in."""
        return self._bounds

    @property
    def lower(self) -> Any:
        """Lower bound."""
        return self._bounds.lower

    @property
    def upper(self) -> Any:
        """Upper bound."""
        return self._bounds.upper

    @property
    def scaling(self) -> Scaling:
        """Driver scaling for this variable."""
        return self._scaling

    @property
    def indices(self) -> list[int] | None:
        """Which elements are free, or ``None`` for all of them."""
        return None if self._indices is None else list(self._indices)

    @property
    def ref(self) -> float:
        """The reference this variable is scaled by, defaulted from its bounds if not given."""
        return self._scaling.ref if self._scaling.ref is not None else self._bounds.magnitude

    def as_kwargs(self, units: str | None) -> dict[str, Any]:
        """Return the keyword arguments OpenMDAO's ``add_design_var`` takes."""
        arguments: dict[str, Any] = {
            "units": units,
            **self._bounds.as_kwargs(),
            **self._scaling.as_kwargs(default_ref=self.ref),
        }
        if self._indices is not None:
            arguments["indices"] = self._indices
        return arguments

    def __repr__(self) -> str:
        """Return a representation naming the interval."""
        return f"OptimizeSpec({self._bounds.describe()})"


class VariableSpec:
    """One entry of ``design_variables`` or ``initial_conditions``.

    Parameters
    ----------
    name : str
        Name of a variable the black box publishes.
    value : float or sequence of float
        Its value. A sequence is a per-node schedule: two entries are interpolated across the
        phase exactly as ``np.linspace`` does in OpenConcept's own run script.
    units : str or None, optional
        Units of the value.
    source : str, optional
        Where the number came from. A number with no provenance is indistinguishable from a
        guess in the report that quotes it.
    optimize : OptimizeSpec or None, optional
        Present makes this variable a design variable the driver may move.
    """

    __slots__ = ("_name", "_optimize", "_source", "_units", "_value")

    ALLOWED = ("value", "units", "source", "optimize")

    def __init__(
        self,
        name: str,
        value: Any,
        units: str | None = None,
        source: str = "",
        optimize: OptimizeSpec | None = None,
    ) -> None:
        self._name = str(name)
        self._value = value
        self._units = units
        self._source = str(source)
        self._optimize = optimize

    @classmethod
    def from_dict(cls, name: str, data: Any, where: str) -> VariableSpec:
        """Build one entry."""
        mapping = _mapping(data, where)
        _reject_unknown(mapping, cls.ALLOWED, where)
        optimize = mapping.get("optimize")
        return cls(
            name=name,
            value=_number_or_vector(_require(mapping, "value", where), f"{where}.value"),
            units=mapping.get("units"),
            source=str(mapping.get("source", "")),
            optimize=None if optimize is None else OptimizeSpec.from_dict(optimize, f"{where}.optimize"),
        )

    @property
    def name(self) -> str:
        """Name of the black-box variable."""
        return self._name

    @property
    def value(self) -> Any:
        """Its value, in :attr:`units`."""
        return self._value

    @property
    def units(self) -> str | None:
        """Units of the value."""
        return self._units

    @property
    def source(self) -> str:
        """Where the number came from."""
        return self._source

    @property
    def optimize(self) -> OptimizeSpec | None:
        """The ``optimize:`` entry, if this variable is free."""
        return self._optimize

    @property
    def is_free(self) -> bool:
        """Whether the driver may move this variable."""
        return self._optimize is not None

    def parameter(self) -> Parameter:
        """Return this entry as a :class:`~cdadt.parameters.Parameter`.

        Only meaningful for a scalar; a per-node schedule is written straight into the box
        rather than routed through a discipline.
        """
        return Parameter(self._name, float(np.atleast_1d(self._value).ravel()[0]), self._units, self._source)

    def __repr__(self) -> str:
        """Return a representation naming the variable and whether it is free."""
        state = "free" if self.is_free else "fixed"
        return f"VariableSpec({self._name!r}, {self._value!r}, {self._units!r}, {state})"


class ConstraintSpec:
    """One constraint on the design.

    Parameters
    ----------
    name : str
        Either a response name any discipline reports -- ``takeoff_field_length``,
        ``climb_throttle`` -- or a raw black-box path. Response names are preferred: they carry
        their own units.
    bounds : Bounds
        Any form ``add_constraint`` accepts.
    units : str or None, optional
        Units the bounds are stated in.
    scaling : Scaling, optional
        Driver scaling. Defaults to the magnitude of the bound, so that constraints of wildly
        different magnitudes -- a field length in thousands of feet and a climb gradient in
        hundredths of a radian -- are comparable to the optimizer.
    indices : sequence of int or None, optional
        Which elements of a vector response are constrained.
    linear : bool, optional
        Whether the constraint is linear in the design variables. Default ``False``.
    regulation, source : str, optional
        Optional provenance. Supplying both is what puts a constraint in the traceability
        matrix as a defensible requirement rather than as a bare number; leaving them out is
        allowed, and the matrix says so.
    title : str, optional
        One line naming the constraint in a report. Defaults to the variable name.
    """

    __slots__ = ("_bounds", "_indices", "_linear", "_name", "_regulation", "_scaling", "_source", "_title", "_units")

    ALLOWED = (
        "name",
        "lower",
        "upper",
        "equals",
        "units",
        "indices",
        "linear",
        "regulation",
        "source",
        "title",
        *Scaling.ALLOWED,
    )

    def __init__(
        self,
        name: str,
        bounds: Bounds,
        units: str | None = None,
        scaling: Scaling | None = None,
        indices: Sequence[int] | None = None,
        linear: bool = False,
        regulation: str = "",
        source: str = "",
        title: str = "",
    ) -> None:
        self._name = str(name)
        self._bounds = bounds
        self._units = units
        self._scaling = scaling if scaling is not None else Scaling()
        self._indices = None if indices is None else [int(index) for index in indices]
        self._linear = bool(linear)
        self._regulation = str(regulation)
        self._source = str(source)
        self._title = str(title) if title else str(name)

    @classmethod
    def from_dict(cls, data: Any, where: str) -> ConstraintSpec:
        """Build one constraint entry."""
        mapping = _mapping(data, where)
        _reject_unknown(mapping, cls.ALLOWED, where)
        try:
            bounds = Bounds(lower=mapping.get("lower"), upper=mapping.get("upper"), equals=mapping.get("equals"))
            scaling = Scaling.from_dict(mapping)
        except ConfigError as error:
            raise ConfigError(f"'{where}': {error}") from None
        return cls(
            name=_require(mapping, "name", where),
            bounds=bounds,
            units=mapping.get("units"),
            scaling=scaling,
            indices=mapping.get("indices"),
            linear=bool(mapping.get("linear", False)),
            regulation=str(mapping.get("regulation", "")),
            source=str(mapping.get("source", "")),
            title=str(mapping.get("title", "")),
        )

    @property
    def name(self) -> str:
        """Response name or black-box path being constrained."""
        return self._name

    @property
    def bounds(self) -> Bounds:
        """The bound imposed."""
        return self._bounds

    @property
    def units(self) -> str | None:
        """Units the bounds are stated in."""
        return self._units

    @property
    def scaling(self) -> Scaling:
        """Driver scaling for this constraint."""
        return self._scaling

    @property
    def indices(self) -> list[int] | None:
        """Which elements of a vector response are constrained."""
        return None if self._indices is None else list(self._indices)

    @property
    def linear(self) -> bool:
        """Whether the constraint is linear in the design variables."""
        return self._linear

    @property
    def regulation(self) -> str:
        """The regulation this constraint comes from, if it was given one."""
        return self._regulation

    @property
    def source(self) -> str:
        """Where the number came from, if it was given."""
        return self._source

    @property
    def title(self) -> str:
        """One line naming the constraint in a report."""
        return self._title

    @property
    def is_traceable(self) -> bool:
        """Whether this constraint names both a regulation and a source."""
        return bool(self._regulation and self._source)

    def __repr__(self) -> str:
        """Return a representation naming the constraint and its bound."""
        return f"ConstraintSpec({self._name!r}, {self._bounds.describe()})"


class ObjectiveSpec:
    """What the optimizer minimizes or maximizes.

    Parameters
    ----------
    name : str
        A response name or a raw black-box path.
    units : str or None, optional
        Units to optimize in.
    scaling : Scaling, optional
        Driver scaling. It matters: SciPy's SLSQP takes its finite-difference step and its
        convergence test on the *scaled* objective, so an objective of order 1e4 is effectively
        converged before it starts.
    sense : str, optional
        ``"minimize"`` or ``"maximize"``. Default ``"minimize"``.
    index : int or None, optional
        Which element of a vector output is the objective.
    """

    __slots__ = ("_index", "_name", "_scaling", "_sense", "_units")

    ALLOWED = ("name", "units", "sense", "index", *Scaling.ALLOWED)
    SENSES = ("minimize", "maximize")

    def __init__(
        self,
        name: str,
        units: str | None = None,
        scaling: Scaling | None = None,
        sense: str = "minimize",
        index: int | None = None,
    ) -> None:
        if sense not in self.SENSES:
            raise ConfigError(f"Objective sense must be one of {list(self.SENSES)}; got {sense!r}.")
        self._name = str(name)
        self._units = units
        self._scaling = scaling if scaling is not None else Scaling()
        self._sense = sense
        self._index = None if index is None else int(index)

    @classmethod
    def from_dict(cls, data: Any) -> ObjectiveSpec:
        """Build from the ``objective`` section."""
        where = "objective"
        mapping = _mapping(data, where)
        _reject_unknown(mapping, cls.ALLOWED, where)
        try:
            scaling = Scaling.from_dict(mapping)
        except ConfigError as error:
            raise ConfigError(f"'{where}': {error}") from None
        return cls(
            name=_require(mapping, "name", where),
            units=mapping.get("units"),
            scaling=scaling,
            sense=str(mapping.get("sense", "minimize")),
            index=mapping.get("index"),
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
    def scaling(self) -> Scaling:
        """Driver scaling for the objective."""
        return self._scaling

    @property
    def sense(self) -> str:
        """``"minimize"`` or ``"maximize"``."""
        return self._sense

    @property
    def index(self) -> int | None:
        """Which element of a vector output is the objective."""
        return self._index

    @property
    def direction(self) -> float:
        """Return ``+1`` to minimize or ``-1`` to maximize."""
        return 1.0 if self._sense == "minimize" else -1.0

    def as_kwargs(self, units: str | None) -> dict[str, Any]:
        """Return the keyword arguments OpenMDAO's ``add_objective`` takes.

        The sense is folded into the scaling, because OpenMDAO drivers only minimize. Both
        spellings are handled, so a case file may scale with ``ref`` or with ``scaler`` and
        still say ``sense: maximize``.
        """
        arguments: dict[str, Any] = {"units": units}
        scaling = self._scaling.as_kwargs(default_ref=1.0)
        if "scaler" in scaling:
            scaling["scaler"] *= self.direction
        else:
            scaling["ref"] = scaling.get("ref", 1.0) * self.direction
            if "ref0" in scaling:
                scaling["ref0"] *= self.direction
        arguments.update(scaling)
        if self._index is not None:
            arguments["index"] = self._index
        return arguments

    def __repr__(self) -> str:
        """Return a representation naming the objective and its sense."""
        return f"ObjectiveSpec({self._name!r}, sense={self._sense!r})"


# =============================================================================================
# Sections
# =============================================================================================


class BlackBoxConfig:
    """Which sizing analysis to drive, and on what grid."""

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
    """How hard to converge the black box. Defaults are OpenConcept's own sizing settings."""

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

    @property
    def maxiter(self) -> int:
        """Newton iteration limit."""
        return self._maxiter

    def __repr__(self) -> str:
        """Return a representation naming the iteration limit."""
        return f"SolverConfig(maxiter={self._maxiter})"


class DriverConfig:
    """Which optimizer to use, and how hard to drive it."""

    __slots__ = ("_derivative_mode", "_maxiter", "_name", "_options", "_tol")

    ALLOWED = ("name", "maxiter", "tol", "derivative_mode", "options")
    MODES = ("auto", "fwd", "rev")

    def __init__(
        self,
        name: str = "SLSQP",
        maxiter: int = 50,
        tol: float = 1e-6,
        derivative_mode: str = "fwd",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if derivative_mode not in self.MODES:
            raise ConfigError(f"derivative_mode must be one of {list(self.MODES)}; got {derivative_mode!r}.")
        self._name = str(name)
        self._maxiter = int(maxiter)
        self._tol = float(tol)
        self._derivative_mode = derivative_mode
        self._options = dict(options or {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriverConfig:
        """Build from the ``driver`` section."""
        _reject_unknown(data, cls.ALLOWED, "driver")
        return cls(
            name=str(data.get("name", "SLSQP")),
            maxiter=int(data.get("maxiter", 50)),
            tol=float(data.get("tol", 1e-6)),
            derivative_mode=str(data.get("derivative_mode", "fwd")),
            options=data.get("options", {}),
        )

    @property
    def name(self) -> str:
        """Optimizer name."""
        return self._name

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
    def options(self) -> Mapping[str, Any]:
        """Extra optimizer settings, passed through to pyOptSparse."""
        return dict(self._options)

    def __repr__(self) -> str:
        """Return a representation naming the optimizer."""
        return f"DriverConfig({self._name!r}, maxiter={self._maxiter})"


# =============================================================================================
# The case file
# =============================================================================================


class Config:
    """A whole case file.

    Examples
    --------
    >>> config = Config.from_yaml("cases/b738.yaml")
    >>> config.black_box.num_nodes
    21
    >>> [name for name, spec in config.design_variables.items() if spec.is_free]
    ['ac|geom|wing|S_ref', ...]
    """

    __slots__ = (
        "_black_box",
        "_constraints",
        "_continuation",
        "_design_variables",
        "_driver",
        "_initial_conditions",
        "_mission_path",
        "_objective",
        "_path",
        "_solver",
    )

    ALLOWED = (
        "black_box",
        "solver",
        "mission_path",
        "design_variables",
        "initial_conditions",
        "continuation",
        "driver",
        "constraints",
        "objective",
    )

    def __init__(
        self,
        black_box: BlackBoxConfig,
        design_variables: Mapping[str, VariableSpec],
        initial_conditions: Mapping[str, VariableSpec] | None = None,
        continuation: ContinuationLadder | None = None,
        solver: SolverConfig | None = None,
        driver: DriverConfig | None = None,
        constraints: Sequence[ConstraintSpec] = (),
        objective: ObjectiveSpec | None = None,
        mission_path: str = "mission",
        path: Path | None = None,
    ) -> None:
        self._black_box = black_box
        self._design_variables = dict(design_variables)
        self._initial_conditions = dict(initial_conditions or {})
        self._continuation = continuation if continuation is not None else ContinuationLadder()
        self._solver = solver if solver is not None else SolverConfig()
        self._driver = driver if driver is not None else DriverConfig()
        self._constraints = tuple(constraints)
        self._objective = objective
        self._mission_path = str(mission_path)
        self._path = path

        duplicated = sorted(set(self._design_variables) & set(self._initial_conditions))
        if duplicated:
            raise ConfigError(
                f"{duplicated} appear in both 'design_variables' and 'initial_conditions'. "
                f"A variable is set once, in one place."
            )
        if self._objective is not None and not self.free_variables:
            raise ConfigError(
                "An objective was given but no variable carries an 'optimize:' entry, so there "
                "is nothing for the driver to move."
            )

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a case file."""
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

        steps = []
        for index, entry in enumerate(_sequence(data.get("continuation", []), "continuation")):
            where = f"continuation[{index}]"
            step = _mapping(entry, where)
            _reject_unknown(step, ("description", "initial_conditions"), where)
            steps.append(
                ContinuationStep(
                    description=str(step.get("description", f"step {index + 1}")),
                    conditions=cls._read_conditions(step.get("initial_conditions", {}), f"{where}.initial_conditions"),
                )
            )

        objective = data.get("objective")
        return cls(
            black_box=BlackBoxConfig.from_dict(_mapping(_require(data, "black_box", "the case file"), "black_box")),
            design_variables=cls._read_variables(
                _require(data, "design_variables", "the case file"), "design_variables"
            ),
            initial_conditions=cls._read_variables(data.get("initial_conditions", {}), "initial_conditions"),
            continuation=ContinuationLadder(steps),
            solver=SolverConfig.from_dict(_mapping(data.get("solver", {}), "solver")),
            driver=DriverConfig.from_dict(_mapping(data.get("driver", {}), "driver")),
            constraints=[
                ConstraintSpec.from_dict(entry, f"constraints[{index}]")
                for index, entry in enumerate(_sequence(data.get("constraints", []), "constraints"))
            ],
            objective=None if objective is None else ObjectiveSpec.from_dict(objective),
            mission_path=str(data.get("mission_path", "mission")),
            path=path,
        )

    @staticmethod
    def _read_variables(entries: Any, where: str) -> dict[str, VariableSpec]:
        """Return ``name -> VariableSpec`` from a mapping of entries."""
        mapping = _mapping(entries, where)
        return {name: VariableSpec.from_dict(name, entry, f"{where}.{name}") for name, entry in mapping.items()}

    @staticmethod
    def _read_conditions(entries: Any, where: str) -> dict[str, tuple[Any, str | None]]:
        """Return ``name -> (value, units)`` from a mapping of entries."""
        mapping = _mapping(entries, where)
        conditions = {}
        for name, entry in mapping.items():
            item = _mapping(entry, f"{where}.{name}")
            _reject_unknown(item, ("value", "units"), f"{where}.{name}")
            conditions[name] = (
                _number_or_vector(_require(item, "value", f"{where}.{name}"), f"{where}.{name}.value"),
                item.get("units"),
            )
        return conditions

    # -- state ---------------------------------------------------------------------------

    @property
    def black_box(self) -> BlackBoxConfig:
        """Which sizing analysis to drive."""
        return self._black_box

    @property
    def solver(self) -> SolverConfig:
        """How hard to converge the box."""
        return self._solver

    @property
    def driver(self) -> DriverConfig:
        """Which optimizer to use."""
        return self._driver

    @property
    def mission_path(self) -> str:
        """Subsystem the mission lives under inside the black box."""
        return self._mission_path

    @property
    def design_variables(self) -> dict[str, VariableSpec]:
        """Every design variable, free or fixed, as ``B738.py`` lists them."""
        return dict(self._design_variables)

    @property
    def initial_conditions_specs(self) -> dict[str, VariableSpec]:
        """Every initial condition, as ``set_values`` writes them."""
        return dict(self._initial_conditions)

    @property
    def free_variables(self) -> dict[str, VariableSpec]:
        """The variables that carry an ``optimize:`` entry, in declaration order."""
        return {
            name: spec
            for name, spec in (*self._design_variables.items(), *self._initial_conditions.items())
            if spec.is_free
        }

    @property
    def constraints(self) -> tuple[ConstraintSpec, ...]:
        """The constraints, in the order the case file states them."""
        return self._constraints

    @property
    def objective(self) -> ObjectiveSpec | None:
        """The objective, or ``None`` for a sizing-only case."""
        return self._objective

    @property
    def continuation(self) -> ContinuationLadder:
        """The ladder walked before the design mission."""
        return self._continuation

    @property
    def path(self) -> Path | None:
        """Where the case file was read from."""
        return self._path

    @property
    def is_optimization(self) -> bool:
        """Whether this case declares an objective to drive."""
        return self._objective is not None

    def initial_conditions(self) -> InitialConditions:
        """Return everything that is written into the box before it is converged.

        Both blocks: the design variables carry values too, and a value is a value whichever
        block it was declared in. Keeping them in separate sections is about what a reader is
        being told, not about what the box is sent.
        """
        values = {
            name: (spec.value, spec.units)
            for name, spec in (*self._design_variables.items(), *self._initial_conditions.items())
        }
        return InitialConditions(values, self._mission_path)

    def parameters(self) -> list[Parameter]:
        """Return the scalar design variables as routable parameters."""
        return [spec.parameter() for spec in self._design_variables.values() if np.atleast_1d(spec.value).size == 1]

    def __repr__(self) -> str:
        """Return a representation naming the file and what it declares."""
        kind = "optimization" if self.is_optimization else "sizing"
        where = str(self._path) if self._path else "in memory"
        return f"Config({where}, {kind}, {len(self._design_variables)} design variables)"
