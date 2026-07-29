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

from collections.abc import Iterator, Mapping, Sequence
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
    "CaseFileSection",
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


class _Absent:
    """The absence of a default, so that ``None`` can be a default like any other value.

    ``__slots__`` is empty deliberately: :data:`ABSENT` is module-level, and a module-level
    object that can be written to is shared mutable state however little it holds.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a representation that reads as what it means in a signature."""
        return "ABSENT"


#: Sentinel meaning "no default was given", which makes the block required.
ABSENT = _Absent()


# =============================================================================================
# Reading the file
# =============================================================================================


class CaseFileSection:
    """One block of a case file, together with the address it lives at.

    Every reader in this module goes through one of these. The point is that a section carries
    its own address, so a message can say *where* in the file the mistake is without every call
    site passing that string along by hand -- and so a nested block cannot be given the wrong
    address, because it derives its own from its parent.

    A case file is the only thing a user of cdadt writes by hand, and it fails late or not at
    all if it is read leniently: a misspelled key means the run succeeds and answers a different
    question. So a section validates on construction, rejects keys it does not recognise, and
    names the block in every message it raises.

    Parameters
    ----------
    data : Any
        The parsed block. Must be a mapping; anything else is the error this raises.
    where : str
        The address, as it should read in a message -- ``"solver"``,
        ``"design_variables.ac|geom|wing|AR"``, ``"constraints[0]"``.

    Raises
    ------
    ConfigError
        If ``data`` is not a mapping, naming ``where`` and what was found instead.

    Examples
    --------
    >>> section = CaseFileSection({"value": 124.6, "units": "m**2"}, "design_variables.S_ref")
    >>> section.require("value")
    124.6
    >>> section.child_of("units")
    Traceback (most recent call last):
    ConfigError: 'design_variables.S_ref.units' must be a mapping; got str.
    """

    __slots__ = ("_data", "_where")

    def __init__(self, data: Any, where: str) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(f"'{where}' must be a mapping; got {type(data).__name__}.")
        self._data = data
        self._where = str(where)

    # -- what it is ----------------------------------------------------------------------

    @property
    def where(self) -> str:
        """The address this section lives at, as it reads in a message."""
        return self._where

    def __contains__(self, key: str) -> bool:
        """Whether the block has ``key``."""
        return key in self._data

    def __repr__(self) -> str:
        """Return a representation naming the address and the keys present."""
        return f"CaseFileSection({self._where!r}, keys={sorted(self._data)})"

    # -- reading -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``key``'s raw value, or ``default`` if the block does not have it."""
        return self._data.get(key, default)

    def require(self, key: str) -> Any:
        """Return ``key``'s raw value.

        Raises
        ------
        ConfigError
            If the block does not have it, naming the block and the key.
        """
        if key not in self._data:
            raise ConfigError(f"'{self._where}' requires a '{key}' key.")
        return self._data[key]

    def reject_unknown(self, allowed: Sequence[str]) -> None:
        """Raise if the block carries any key outside ``allowed``.

        Raises
        ------
        ConfigError
            Naming the offending keys and listing what is allowed. A silently ignored key means
            the study runs and answers a different question than the one that was written.
        """
        unknown = sorted(set(self._data) - set(allowed))
        if unknown:
            raise ConfigError(f"'{self._where}' has unknown key(s) {unknown}. Allowed keys are {sorted(allowed)}.")

    def number_or_vector(self, key: str) -> Any:
        """Return ``key`` as a float, or an array when the file gave a sequence."""
        return self.as_number_or_vector(self.require(key), f"{self._where}.{key}")

    # -- descending ----------------------------------------------------------------------

    def child_of(self, key: str, default: Any = ABSENT, *, where: str | None = None) -> CaseFileSection:
        """Return the block at ``key`` as a section of its own.

        Parameters
        ----------
        key : str
            Which block to descend into.
        default : Any, optional
            Used when the key is absent, so an optional block reads the same as a present one --
            pass ``{}`` for a block whose keys all have defaults of their own. Omit it to make
            the block required, in which case an absent key raises naming this section.
        where : str, optional
            Address to give the child. Defaults to ``"<this section>.<key>"``, which is what a
            nested block wants. The top-level blocks of a case file pass their own bare name
            instead: a reader is looking for ``'objective'`` in the message, not
            ``'the case file.objective'``.
        """
        if key not in self._data and default is ABSENT:
            self.require(key)
        raw = self._data.get(key, default)
        return CaseFileSection(raw, where if where is not None else f"{self._where}.{key}")

    def entries(self) -> Iterator[tuple[str, CaseFileSection]]:
        """Iterate ``(name, section)`` for a block whose keys are themselves blocks.

        This is what ``design_variables`` and ``initial_conditions`` are: a mapping from a
        black-box variable name to its entry.
        """
        for name in self._data:
            yield name, CaseFileSection(self._data[name], f"{self._where}.{name}")

    def series(self, key: str) -> tuple[CaseFileSection, ...]:
        """Return the list at ``key`` as sections addressed ``key[0]``, ``key[1]``, ...

        Absent means empty, because a case file that constrains nothing simply omits the block.

        Raises
        ------
        ConfigError
            If the value is present but is not a list.
        """
        raw = self._data.get(key, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ConfigError(f"'{key}' must be a list; got {type(raw).__name__}.")
        return tuple(CaseFileSection(entry, f"{key}[{index}]") for index, entry in enumerate(raw))

    # -- converting a value that did not come from a key ---------------------------------

    @staticmethod
    def as_number(value: Any, where: str) -> float:
        """Return ``value`` as a float, or raise naming ``where`` it came from."""
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"'{where}' must be a number; got {value!r}.") from error

    @staticmethod
    def as_number_or_vector(value: Any, where: str) -> Any:
        """Return a float, or an array when the case file gave a sequence.

        A sequence is how a per-node schedule is written -- ``[2300, 400]`` for the endpoints of
        a climb rate -- and how a vector design variable is bounded element by element, which is
        what OpenConcept's aerostructural example does for a spanwise thickness distribution.
        """
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray([float(item) for item in value])
            except (TypeError, ValueError) as error:
                raise ConfigError(f"'{where}' has a non-numeric entry: {value!r}.") from error
        return CaseFileSection.as_number(value, where)


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
    def from_section(cls, section: CaseFileSection) -> Scaling:
        """Build from whichever of the four keys an entry carries.

        Scaling shares its block with whatever else the entry declares -- bounds, indices, a
        regulation -- so this reads the four keys it owns and leaves the rest to the caller.
        """
        return cls(**{key: section.get(key) for key in cls.ALLOWED if key in section})

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
        read = CaseFileSection.as_number_or_vector
        self._lower = None if lower is None else read(lower, "constraint lower bound")
        self._upper = None if upper is None else read(upper, "constraint upper bound")
        self._equals = None if equals is None else read(equals, "constraint equality")
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
            """Render one side of the bound, summarising a vector rather than printing it."""
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
    def from_section(cls, section: CaseFileSection) -> OptimizeSpec:
        """Build one ``optimize:`` entry."""
        section.reject_unknown(cls.ALLOWED)
        try:
            return cls(
                bounds=Bounds(lower=section.get("lower"), upper=section.get("upper")),
                scaling=Scaling.from_section(section),
                indices=section.get("indices"),
            )
        except ConfigError as error:
            raise ConfigError(f"'{section.where}': {error}") from None

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
    def from_section(cls, name: str, section: CaseFileSection) -> VariableSpec:
        """Build one entry."""
        section.reject_unknown(cls.ALLOWED)
        return cls(
            name=name,
            value=section.number_or_vector("value"),
            units=section.get("units"),
            source=str(section.get("source", "")),
            optimize=None if "optimize" not in section else OptimizeSpec.from_section(section.child_of("optimize")),
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
    def from_section(cls, section: CaseFileSection) -> ConstraintSpec:
        """Build one constraint entry."""
        section.reject_unknown(cls.ALLOWED)
        try:
            bounds = Bounds(lower=section.get("lower"), upper=section.get("upper"), equals=section.get("equals"))
            scaling = Scaling.from_section(section)
        except ConfigError as error:
            raise ConfigError(f"'{section.where}': {error}") from None
        return cls(
            name=section.require("name"),
            bounds=bounds,
            units=section.get("units"),
            scaling=scaling,
            indices=section.get("indices"),
            linear=bool(section.get("linear", False)),
            regulation=str(section.get("regulation", "")),
            source=str(section.get("source", "")),
            title=str(section.get("title", "")),
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
    def from_section(cls, section: CaseFileSection) -> ObjectiveSpec:
        """Build from the ``objective`` section."""
        section.reject_unknown(cls.ALLOWED)
        try:
            scaling = Scaling.from_section(section)
        except ConfigError as error:
            raise ConfigError(f"'{section.where}': {error}") from None
        return cls(
            name=section.require("name"),
            units=section.get("units"),
            scaling=scaling,
            sense=str(section.get("sense", "minimize")),
            index=section.get("index"),
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
    def from_section(cls, section: CaseFileSection) -> BlackBoxConfig:
        """Build from the ``black_box`` section."""
        section.reject_unknown(cls.ALLOWED)
        return cls(model=section.require("model"), num_nodes=section.require("num_nodes"))

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
    def from_section(cls, section: CaseFileSection) -> SolverConfig:
        """Build from the ``solver`` section, which may be empty."""
        section.reject_unknown(cls.ALLOWED)
        return cls(**{key: section.get(key) for key in cls.ALLOWED if key in section})

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
    def from_section(cls, section: CaseFileSection) -> DriverConfig:
        """Build from the ``driver`` section."""
        section.reject_unknown(cls.ALLOWED)
        return cls(
            name=str(section.get("name", "SLSQP")),
            maxiter=int(section.get("maxiter", 50)),
            tol=float(section.get("tol", 1e-6)),
            derivative_mode=str(section.get("derivative_mode", "fwd")),
            options=section.get("options", {}),
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
        """Build a configuration from an already-parsed mapping.

        This is where the raw document becomes a :class:`CaseFileSection`; every reader below
        this point takes a section and therefore knows its own address in the file.
        """
        return cls.from_section(CaseFileSection(data, "the case file"), path=path)

    @classmethod
    def from_section(cls, section: CaseFileSection, path: Path | None = None) -> Config:
        """Build a configuration from the top-level section of a case file."""
        section.reject_unknown(cls.ALLOWED)

        steps = []
        for index, rung in enumerate(section.series("continuation")):
            rung.reject_unknown(("description", "initial_conditions"))
            steps.append(
                ContinuationStep(
                    description=str(rung.get("description", f"step {index + 1}")),
                    conditions=cls._read_conditions(rung.child_of("initial_conditions", {})),
                )
            )

        # The top-level blocks are addressed by their own bare names, so that a message points at
        # 'objective' the way the file spells it rather than at 'the case file.objective'.
        return cls(
            black_box=BlackBoxConfig.from_section(section.child_of("black_box", where="black_box")),
            design_variables=cls._read_variables(section.child_of("design_variables", where="design_variables")),
            initial_conditions=cls._read_variables(
                section.child_of("initial_conditions", {}, where="initial_conditions")
            ),
            continuation=ContinuationLadder(steps),
            solver=SolverConfig.from_section(section.child_of("solver", {}, where="solver")),
            driver=DriverConfig.from_section(section.child_of("driver", {}, where="driver")),
            constraints=[ConstraintSpec.from_section(entry) for entry in section.series("constraints")],
            objective=(
                None
                if "objective" not in section
                else ObjectiveSpec.from_section(section.child_of("objective", where="objective"))
            ),
            mission_path=str(section.get("mission_path", "mission")),
            path=path,
        )

    @staticmethod
    def _read_variables(section: CaseFileSection) -> dict[str, VariableSpec]:
        """Return ``name -> VariableSpec`` from a block of entries."""
        return {name: VariableSpec.from_section(name, entry) for name, entry in section.entries()}

    @staticmethod
    def _read_conditions(section: CaseFileSection) -> dict[str, tuple[Any, str | None]]:
        """Return ``name -> (value, units)`` from a block of entries."""
        conditions = {}
        for name, item in section.entries():
            item.reject_unknown(("value", "units"))
            conditions[name] = (
                item.number_or_vector("value"),
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
