"""The certification basis: what must hold, where it comes from, and whether it does.

This is the module the tool is named for. A certification-driven design is one where the
requirements are not commentary on the answer but constraints on the search, and where every
one of them can be traced from a regulation, through a number with a stated source, to the
quantity that was actually evaluated.

Each requirement is a class. It knows which regulation it comes from, which response of the
black box tests it, which way the inequality runs, and what units it is stated in. The *limit*
and its *source* are constructor arguments, never defaults: which runway, which aerodrome, which
aeroplane class and which engine count belong to the operating case, not to the code. A
requirement that cannot name its regulation and its source is a constraint nobody can defend,
so both are required.

What can be constrained, and what cannot
----------------------------------------

Every requirement here is a function of quantities the black box actually publishes. That is a
real limit, and it is the honest one: cdadt does not compute physics, so it cannot enforce a
regulation whose governing quantity the box does not produce. Reference landing approach speed
is the clearest example -- it needs the reference stall speed at maximum landing weight, which
the box does not publish, and inventing a stall-speed model here to close the gap would be
exactly the reimplementation the black-box boundary exists to prevent. :doc:`/certification`
lists what is and is not reachable, and :doc:`/validation` states the consequence.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import openmdao.api as om

from cdadt.config import RequirementSpec
from cdadt.results import ResponseCatalog

__all__ = [
    "SHIPPED_REQUIREMENTS",
    "BalancedFieldLength",
    "CertificationBasis",
    "DesignRange",
    "EngineOutClimbGradient",
    "MaximumTakeoffWeight",
    "Requirement",
    "RequirementCatalog",
    "RequirementError",
    "RequirementResult",
    "ResponseLimit",
    "ThrottleLimit",
]


class RequirementError(Exception):
    """Raised when a requirement cannot be built or evaluated as stated."""


class RequirementResult:
    """One requirement, evaluated against one converged design.

    Parameters
    ----------
    requirement : Requirement
        What was checked.
    value : float
        The governing value: the largest node for an upper limit, the smallest for a lower one.
        A vector response such as a throttle history is one requirement, not twenty-one, and the
        node that governs is the one that decides it.

    Attributes
    ----------
    ACTIVE_TOLERANCE : float
        Relative distance from the limit within which a requirement is called active rather than
        merely met. An active requirement is one that shaped the design.
    """

    ACTIVE_TOLERANCE: ClassVar[float] = 1e-4

    __slots__ = ("_requirement", "_value")

    def __init__(self, requirement: Requirement, value: float) -> None:
        self._requirement = requirement
        self._value = float(value)

    @property
    def requirement(self) -> Requirement:
        """The requirement that was checked."""
        return self._requirement

    @property
    def value(self) -> float:
        """The governing value, in the requirement's units."""
        return self._value

    @property
    def margin(self) -> float:
        """How far the design is on the satisfying side of the limit.

        Positive is compliant. For an upper limit this is ``limit - value``; for a lower limit
        it is ``value - limit``, so that the sign means the same thing either way.
        """
        limit = self._requirement.limit
        return limit - self._value if self._requirement.sense == "upper" else self._value - limit

    @property
    def relative_margin(self) -> float:
        """The margin as a fraction of the limit, or ``inf`` for a limit of zero."""
        limit = abs(self._requirement.limit)
        return self.margin / limit if limit else float("inf")

    @property
    def satisfied(self) -> bool:
        """Whether the requirement is met."""
        return self.margin >= -self.ACTIVE_TOLERANCE * max(abs(self._requirement.limit), 1.0)

    @property
    def active(self) -> bool:
        """Whether the design sits on the limit, and was therefore shaped by it."""
        return self.satisfied and abs(self.relative_margin) <= self.ACTIVE_TOLERANCE

    @property
    def status(self) -> str:
        """``"MET"``, ``"ACTIVE"`` or ``"VIOLATED"``."""
        if not self.satisfied:
            return "VIOLATED"
        return "ACTIVE" if self.active else "MET"

    def __repr__(self) -> str:
        """Return a representation naming the requirement and its status."""
        return f"RequirementResult({self._requirement.name!r}, {self._value:.4f}, {self.status})"


class Requirement:
    """One requirement the design must satisfy.

    Subclasses fix the physics -- which response tests the requirement and which way the
    inequality runs -- and leave the number to the case file.

    Parameters
    ----------
    limit : float
        The limiting value, in ``units``.
    regulation : str
        The regulation or standard the limit comes from, e.g. ``"14 CFR 25.113"``. Use
        ``"design"`` for a requirement that is a programme decision rather than a rule, and say
        so; a design decision presented as a regulation is worse than no citation.
    source : str
        Where this particular number comes from: which runway, which aerodrome category, which
        aeroplane class.
    units : str or None, optional
        Units the limit is stated in. Defaults to the response's own units, which is nearly
        always what is wanted; give it when the case file states the limit in something else.
    options : mapping, optional
        Subclass-specific settings, such as which phase a throttle limit applies to.

    Attributes
    ----------
    kind : str
        The name a case file uses in ``type:``. A :class:`RequirementCatalog` maps it to the
        class.
    response : str
        The catalogued response this requirement is evaluated on.
    sense : str
        ``"upper"`` if the response must not exceed the limit, ``"lower"`` if it must not fall
        below it.
    title : str
        One line naming the requirement in a traceability matrix.

    Notes
    -----
    There is deliberately no class-level registry here, and no ``__init_subclass__`` that writes
    to one. A registry populated at import time is shared mutable state: two studies in one
    process share it, defining a class anywhere mutates it, and what a case file resolves to
    depends on what has been imported. :class:`RequirementCatalog` holds the same mapping as
    instance state instead, so a study's set of available requirements is something it owns and
    can be handed a different value of.
    """

    kind: ClassVar[str] = ""
    response: ClassVar[str] = ""
    sense: ClassVar[str] = "upper"
    title: ClassVar[str] = ""

    #: Option keys this requirement understands, beyond the shared ones.
    option_keys: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        limit: float,
        regulation: str,
        source: str,
        units: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if self.sense not in ("upper", "lower"):
            raise RequirementError(f"'{type(self).__name__}.sense' must be 'upper' or 'lower'.")
        if not regulation:
            raise RequirementError(f"Requirement '{self.kind}' must name the regulation its limit comes from.")
        if not source:
            raise RequirementError(f"Requirement '{self.kind}' must record where its limit came from.")
        self._limit = float(limit)
        self._regulation = str(regulation)
        self._source = str(source)
        self._units = units
        self._options = dict(options or {})

        unknown = sorted(set(self._options) - set(self.option_keys))
        if unknown:
            raise RequirementError(
                f"Requirement '{self.kind}' does not understand the option(s) {unknown}; "
                f"it understands {list(self.option_keys)}."
            )

    # -- construction --------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: RequirementSpec, catalog: RequirementCatalog | None = None) -> Requirement:
        """Build the requirement a case file's entry describes.

        Parameters
        ----------
        spec : RequirementSpec
            The case file's entry.
        catalog : RequirementCatalog, optional
            Which requirement classes are available. Defaults to a fresh catalogue of the
            shipped ones -- fresh, rather than a shared instance, so that nothing a caller does
            to it can affect anybody else.

        Raises
        ------
        RequirementError
            If no requirement class answers to that ``type``.
        """
        return (catalog if catalog is not None else RequirementCatalog()).build(spec)

    # -- state ---------------------------------------------------------------------------

    @property
    def limit(self) -> float:
        """The limiting value."""
        return self._limit

    @property
    def regulation(self) -> str:
        """The regulation the limit comes from."""
        return self._regulation

    @property
    def source(self) -> str:
        """Where this particular number came from."""
        return self._source

    @property
    def options(self) -> Mapping[str, Any]:
        """Subclass-specific settings."""
        return dict(self._options)

    @property
    def name(self) -> str:
        """A unique name for this requirement instance, used as its constraint alias."""
        return self.kind

    def response_name(self) -> str:
        """Return the catalogued response this requirement is evaluated on."""
        return self.response

    def units(self, catalog: ResponseCatalog) -> str | None:
        """Return the units this requirement is stated in."""
        return self._units if self._units is not None else catalog.units(self.response_name())

    def path(self, catalog: ResponseCatalog) -> str:
        """Return the black-box path of the response this requirement tests."""
        return catalog.path(self.response_name())

    # -- use -----------------------------------------------------------------------------

    def register(self, model: om.Group, catalog: ResponseCatalog) -> None:
        """Declare this requirement as a constraint on the black box's group.

        Parameters
        ----------
        model : openmdao.api.Group
            The box's group, before ``setup``.
        catalog : ResponseCatalog
            Used to turn the response name into a path and its units.

        Notes
        -----
        The constraint is scaled by the limit so that requirements of wildly different
        magnitudes -- a field length in thousands of feet and a climb gradient in hundredths of
        a radian -- are comparable to the optimizer. Without it the gradient constraint is
        numerically invisible next to the field length.
        """
        bound = {"upper" if self.sense == "upper" else "lower": self._limit}
        model.add_constraint(
            self.path(catalog),
            units=self.units(catalog),
            ref=abs(self._limit) or 1.0,
            alias=self.name,
            **bound,
        )

    def evaluate(self, box: Any, catalog: ResponseCatalog) -> RequirementResult:
        """Read the response and return the result.

        A vector response is reduced to the node that governs: the largest for an upper limit,
        the smallest for a lower one.
        """
        value = np.asarray(box.get(self.path(catalog), units=self.units(catalog)))
        governing = float(value.max() if self.sense == "upper" else value.min())
        return RequirementResult(self, governing)

    def __repr__(self) -> str:
        """Return a representation naming the requirement, its sense and its limit."""
        return f"{type(self).__name__}({self.sense} {self._limit}, {self._regulation!r})"


# =============================================================================================
# The shipped requirements
# =============================================================================================


class BalancedFieldLength(Requirement):
    """The balanced field length must fit the runway available.

    14 CFR 25.113 defines the takeoff distance as the greater of the all-engines and
    engine-inoperative cases. The black box solves the decision speed V\\ :sub:`1` implicitly so
    that continuing after a failure at V\\ :sub:`1` and rejecting the takeoff cover the same
    distance, which is the balanced field length; at convergence the continue and abort
    distances are equal, and the validation suite checks that they are.

    The *limit* is not a regulation. It is the runway the operator intends to use, at the
    altitude and temperature intended, and the ``source`` must say which.
    """

    kind: ClassVar[str] = "balanced_field_length"
    response: ClassVar[str] = "takeoff_field_length"
    sense: ClassVar[str] = "upper"
    title: ClassVar[str] = "Balanced field length within the runway available"


class EngineOutClimbGradient(Requirement):
    """The engine-out climb gradient must meet the second-segment minimum.

    14 CFR 25.121(b) requires a positive steady gradient of climb with the critical engine
    inoperative at V\\ :sub:`2`: 2.4% for a two-engine aeroplane, 2.7% for three engines, 3.0%
    for four. Which one applies is a property of the aeroplane, so it is the case file's limit
    and the ``source`` must say which class it is for.

    .. warning::

       The black box evaluates its engine-out climb condition in the **clean** configuration.
       §25.121(b) specifies the takeoff flap setting with the landing gear retracted, which
       produces more drag and therefore a lower gradient. The number this requirement constrains
       is consequently optimistic against the regulation as written. cdadt does not correct it,
       because correcting it would mean changing what the box computes. :doc:`/validation`
       records this as a known gap.
    """

    kind: ClassVar[str] = "engine_out_climb_gradient"
    response: ClassVar[str] = "engine_out_climb_gradient"
    sense: ClassVar[str] = "lower"
    title: ClassVar[str] = "OEI second-segment climb gradient"


class ThrottleLimit(Requirement):
    """Throttle must stay within the range the engine deck is fitted over.

    Not a regulation, and it must not be presented as one: it is the condition under which the
    propulsion model inside the black box is meaningful. The engine is a scaled deck, and asking
    it for more than its rated throttle extrapolates a surrogate rather than describing an
    engine. In practice this constraint is frequently the active one, which is worth knowing.

    Options
    -------
    phase : str
        Which phase's throttle history to constrain: ``"climb"``, ``"cruise"`` or
        ``"descent"``. The whole history is constrained, and the governing node is reported.
    """

    kind: ClassVar[str] = "throttle_limit"
    sense: ClassVar[str] = "upper"
    option_keys: ClassVar[tuple[str, ...]] = ("phase",)
    # ``title`` is a property below: several throttle limits coexist in one basis, and rows in
    # a traceability matrix that read identically are rows nobody can act on.

    def __init__(
        self,
        limit: float,
        regulation: str,
        source: str,
        units: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(limit, regulation, source, units, options)
        if "phase" not in self._options:
            raise RequirementError("A throttle limit must say which 'phase' it applies to.")

    @property
    def phase(self) -> str:
        """The phase whose throttle history is constrained."""
        return str(self._options["phase"])

    @property
    def name(self) -> str:
        """A name unique per phase, so several throttle limits can coexist."""
        return f"{self.kind}_{self.phase}"

    @property
    def title(self) -> str:
        """One line naming which phase's throttle is limited."""
        return f"Throttle within the engine deck's range in {self.phase}"

    def response_name(self) -> str:
        """Return the throttle response of the configured phase."""
        return f"{self.phase}_throttle"


class MaximumTakeoffWeight(Requirement):
    """Maximum takeoff weight must not exceed the structural limit certificated.

    A type-certificate limit rather than a design goal: the airframe, the gear and the field
    performance are all certificated at a stated maximum weight, and a sizing loop that closes
    above it has designed a different aeroplane.
    """

    kind: ClassVar[str] = "maximum_takeoff_weight"
    response: ClassVar[str] = "MTOW"
    sense: ClassVar[str] = "upper"
    title: ClassVar[str] = "Maximum takeoff weight within the structural limit"


class DesignRange(Requirement):
    """The design mission must actually be flown.

    The black box solves the cruise duration so that the requested range is reached, so this is
    normally satisfied by construction. It is worth constraining anyway: it is the requirement
    that makes the fuel number mean something, and if a solve ever settles somewhere else this
    is what says so.
    """

    kind: ClassVar[str] = "design_range"
    response: ClassVar[str] = "mission_range_flown"
    sense: ClassVar[str] = "lower"
    title: ClassVar[str] = "Design range flown"


class ResponseLimit(Requirement):
    """A limit on any response, for requirements cdadt does not ship a class for.

    The escape hatch. A study that needs to bound a quantity no shipped requirement covers can
    state it directly in the case file rather than waiting for a class:

    .. code-block:: yaml

        - type: response_limit
          response: total_fuel
          sense: upper
          limit: 20000
          units: kg
          regulation: design
          source: Fuel tank capacity of the wing box as laid out

    It still requires a regulation and a source, for the same reason everything else does. Use a
    named class where one exists: the named classes cannot get the sense or the response wrong,
    and this one can.

    Options
    -------
    response : str
        The catalogued response to constrain.
    sense : str
        ``"upper"`` or ``"lower"``.
    """

    kind: ClassVar[str] = "response_limit"
    option_keys: ClassVar[tuple[str, ...]] = ("response", "sense")
    # ``title`` is a property below rather than a class attribute, because what this
    # requirement is called depends on what the case file pointed it at.

    def __init__(
        self,
        limit: float,
        regulation: str,
        source: str,
        units: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        settings = dict(options or {})
        if "response" not in settings:
            raise RequirementError("A response_limit must say which 'response' it constrains.")
        self.sense = str(settings.get("sense", "upper"))  # instance attribute shadows the class default
        super().__init__(limit, regulation, source, units, settings)

    @property
    def name(self) -> str:
        """A name unique per constrained response."""
        return f"{self.kind}_{self._options['response']}"

    @property
    def title(self) -> str:  # type: ignore[override]
        """One line naming what is limited."""
        direction = "at most" if self.sense == "upper" else "at least"
        return f"{self._options['response']} {direction} the stated limit"

    def response_name(self) -> str:
        """Return the response named in the case file."""
        return str(self._options["response"])


# =============================================================================================
# The catalogue of available requirement types
# =============================================================================================

#: The requirement classes cdadt ships, in the order a reference table lists them. An immutable
#: tuple rather than a mutable registry: it is read to build a catalogue and never written.
SHIPPED_REQUIREMENTS: tuple[type[Requirement], ...] = (
    BalancedFieldLength,
    EngineOutClimbGradient,
    ThrottleLimit,
    MaximumTakeoffWeight,
    DesignRange,
    ResponseLimit,
)


class RequirementCatalog:
    """Which requirement types a study may name in its case file.

    Holds the mapping from the ``type:`` a case file writes to the class that implements it.
    Deliberately an *object* rather than a class-level registry: a registry populated at import
    time is shared mutable state, and it makes what a case file resolves to depend on what has
    happened to be imported. A catalogue is owned by the study that uses it, can be handed a
    different set of classes, and cannot be perturbed from a distance.

    Parameters
    ----------
    requirements : sequence of type, optional
        Requirement classes to offer. Defaults to :data:`SHIPPED_REQUIREMENTS`. Extending is a
        matter of passing a longer sequence -- see :doc:`/tutorials`.

    Raises
    ------
    RequirementError
        If a class declares no ``kind``, or if two declare the same one. The second is what a
        registry would have resolved silently by letting the later class win.

    Examples
    --------
    >>> catalog = RequirementCatalog()
    >>> sorted(catalog)
    ['balanced_field_length', 'design_range', 'engine_out_climb_gradient', ...]
    >>> catalog = RequirementCatalog([*SHIPPED_REQUIREMENTS, MyRequirement])
    """

    def __init__(self, requirements: Sequence[type[Requirement]] | None = None) -> None:
        classes = tuple(requirements) if requirements is not None else SHIPPED_REQUIREMENTS
        by_kind: dict[str, type[Requirement]] = {}
        for requirement_class in classes:
            kind = getattr(requirement_class, "kind", "")
            if not kind:
                raise RequirementError(
                    f"{requirement_class.__name__} declares no 'kind', so no case file could name it."
                )
            if kind in by_kind and by_kind[kind] is not requirement_class:
                raise RequirementError(
                    f"Two requirement classes both call themselves '{kind}': "
                    f"{by_kind[kind].__name__} and {requirement_class.__name__}."
                )
            by_kind[kind] = requirement_class
        self._by_kind = by_kind

    def __contains__(self, kind: str) -> bool:
        """Return whether a requirement of that ``type`` is available."""
        return kind in self._by_kind

    def __iter__(self) -> Iterator[str]:
        """Iterate the available ``type`` names."""
        return iter(self._by_kind)

    def __len__(self) -> int:
        """Return how many requirement types are available."""
        return len(self._by_kind)

    def requirement_class(self, kind: str) -> type[Requirement]:
        """Return the class a case file's ``type`` names.

        Raises
        ------
        RequirementError
            Listing what is available, so a typo does not require reading the source to fix.
        """
        try:
            return self._by_kind[kind]
        except KeyError:
            raise RequirementError(
                f"'{kind}' is not a requirement cdadt knows. Available: {sorted(self._by_kind)}."
            ) from None

    def build(self, spec: RequirementSpec) -> Requirement:
        """Build the requirement a case file's entry describes."""
        return self.requirement_class(spec.kind)(
            limit=spec.limit,
            regulation=spec.regulation,
            source=spec.source,
            units=spec.units,
            options=spec.options,
        )

    def __repr__(self) -> str:
        """Return a representation naming how many types are available."""
        return f"RequirementCatalog({len(self._by_kind)} types)"


# =============================================================================================
# The basis
# =============================================================================================


class CertificationBasis:
    """The set of requirements a design is certified against.

    Parameters
    ----------
    requirements : sequence of Requirement
        What must hold. May be empty: an unconstrained study is a valid thing to run, and the
        report says so rather than printing an empty table.

    Raises
    ------
    RequirementError
        If two requirements would produce the same constraint name, which would silently
        replace one with the other.

    Examples
    --------
    >>> basis = CertificationBasis.from_specs(config.optimization.requirements)
    >>> print(basis.traceability_matrix(box, catalog))
    """

    def __init__(self, requirements: Sequence[Requirement] = ()) -> None:
        self._requirements = tuple(requirements)
        names = [requirement.name for requirement in self._requirements]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise RequirementError(f"More than one requirement is named {duplicates}.")

    @classmethod
    def from_specs(
        cls, specs: Sequence[RequirementSpec], catalog: RequirementCatalog | None = None
    ) -> CertificationBasis:
        """Build the basis a case file's ``requirements`` list describes.

        Parameters
        ----------
        specs : sequence of RequirementSpec
            The case file's ``requirements`` entries.
        catalog : RequirementCatalog, optional
            Which requirement types are available. Defaults to the shipped ones. Pass a wider
            catalogue to make a study's own requirement classes nameable in its case file.
        """
        catalog = catalog if catalog is not None else RequirementCatalog()
        return cls([catalog.build(spec) for spec in specs])

    def __len__(self) -> int:
        """Return the number of requirements."""
        return len(self._requirements)

    def __iter__(self):
        """Iterate the requirements."""
        return iter(self._requirements)

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        """The requirements, in the order the case file states them."""
        return self._requirements

    def register(self, model: om.Group, catalog: ResponseCatalog) -> None:
        """Declare every requirement as a constraint, before setup."""
        for requirement in self._requirements:
            requirement.register(model, catalog)

    def evaluate(self, box: Any, catalog: ResponseCatalog) -> list[RequirementResult]:
        """Evaluate every requirement against a converged design."""
        return [requirement.evaluate(box, catalog) for requirement in self._requirements]

    def traceability_matrix(self, box: Any, catalog: ResponseCatalog) -> str:
        """Return the requirements, their sources and their margins, as a table.

        This is the artefact the whole module exists to produce: every requirement, the
        regulation behind it, the number and where that number came from, the value actually
        achieved, and whether it is met, active or violated.
        """
        if not self._requirements:
            return "No certification basis was attached; nothing was constrained."

        results = self.evaluate(box, catalog)
        titles = max(len(result.requirement.title) for result in results)
        regulations = max(len(result.requirement.regulation) for result in results)
        header = (
            f"{'regulation':<{regulations}s}  {'requirement':<{titles}s}  {'value':>13s} {'limit':>13s} "
            f"{'margin':>13s}  {'units':<6s} status"
        )
        lines = [header, "-" * len(header)]
        for result in results:
            requirement = result.requirement
            units = requirement.units(catalog) or "-"
            lines.append(
                f"{requirement.regulation:<{regulations}s}  {requirement.title:<{titles}s}  "
                f"{result.value:13.4f} {requirement.limit:13.4f} {result.margin:13.4f}  "
                f"{units:<6.6s} {result.status}"
            )
        lines.append("")
        lines.append("Where each limit came from")
        lines.append("--------------------------")
        for result in results:
            requirement = result.requirement
            lines.append(f"  {requirement.name} ({requirement.regulation}): {requirement.source}")

        violated = [r for r in results if not r.satisfied]
        active = [r for r in results if r.active]
        lines.append("")
        lines.append(
            f"{len(results) - len(violated)} of {len(results)} requirements met, "
            f"{len(active)} active, {len(violated)} violated."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation naming the requirement count."""
        return f"CertificationBasis({len(self._requirements)} requirements)"
