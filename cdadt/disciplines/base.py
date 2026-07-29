"""The discipline abstraction.

A discipline is one engineering domain -- geometry, aerodynamics, propulsion, stability,
structures, weights, performance -- modeled as a class. Each one owns two things and nothing
else:

**The parameters it puts into the black box.** Which ``ac|`` variables belong to its domain,
their values, their units and their provenance. Ownership is declared as a pattern and is
checked to be *total and disjoint*: every settable variable the black box publishes is owned by
exactly one discipline, and no variable is owned by two. That check runs against the live model,
not against a list written down here, so adding a variable to the black box surfaces as a
failing test rather than as a silently unowned input.

**The responses it reads back out.** Which quantities the box produces belong to its domain,
where they live inside the box, and in what units to read them.

**What a discipline is not.** It does not compute physics. It has no ``compute``, no residual
and no partial derivative, because the aerodynamics, the weight correlations, the engine deck,
the tail sizing and the whole mission are computed *inside* the black box by OpenConcept, used
as published. A cdadt discipline is the encapsulated, validated interface to its slice of that
box -- the object that knows what its domain is allowed to set, what its domain reports, and
nothing about how either is calculated. Any documentation that implies otherwise is wrong.

There is no global state anywhere in this package. Everything a discipline holds is instance
state reached through properties, which is what lets an optimizer hold several aircraft at once
and what keeps a discipline testable without building a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from fnmatch import fnmatchcase
from typing import Any, ClassVar

from cdadt.parameters import Parameter, Response

__all__ = ["Discipline", "DisciplineError"]


class DisciplineError(Exception):
    """Raised when a discipline is asked to hold something that is not its own."""


class Discipline(ABC):
    """Abstract base for an engineering domain's slice of the black-box interface.

    Subclasses declare three class attributes and add nothing else unless their domain needs
    it:

    ``discipline_name``
        Short identifier used as the key in reports and in the results object.
    ``owned_patterns``
        Shell-style patterns matching the black-box variables this discipline may set. A
        discipline that sets nothing declares an empty tuple -- see
        :class:`~cdadt.disciplines.structures.Structures`, which reports a weight breakdown the
        box computes but has no input of its own.
    ``reported``
        The :class:`~cdadt.parameters.Response` objects this discipline reads back.

    Examples
    --------
    >>> from cdadt.disciplines import Aerodynamics
    >>> aero = Aerodynamics()
    >>> aero.add(Parameter("ac|aero|polar|e", 0.801, source="B738 example estimate"))
    >>> aero.value("ac|aero|polar|e")
    0.801
    >>> aero.owns("ac|geom|wing|S_ref")
    False
    """

    @property
    @abstractmethod
    def discipline_name(self) -> str:
        """Short identifier for the discipline. Unique across the disciplines of one aircraft.

        Abstract because it is the one thing every domain must supply and none can inherit: it
        is the key the discipline is reported and looked up under. Subclasses satisfy it by
        assigning a plain class attribute -- ``discipline_name = "geometry"`` -- which is what
        makes it readable on the class as well as on an instance, and that in turn is what lets
        :class:`~cdadt.aircraft.Aircraft` compose discipline *classes* rather than instances.
        """

    #: One line saying what the domain covers, used in generated documentation and reports.
    description: ClassVar[str] = ""

    #: Shell-style patterns matching the black-box variables this discipline may set.
    owned_patterns: ClassVar[tuple[str, ...]] = ()

    #: Quantities this discipline reads back out of the black box.
    reported: ClassVar[tuple[Response, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require a subclass to name its domain, at the moment the class is written.

        A subclass that does not declare ``discipline_name`` inherits ``"discipline"``. That is
        not harmless: it is the key a discipline is reported and looked up under, so two
        forgetful subclasses would collide, and the failure would surface far away as
        :class:`~cdadt.aircraft.AircraftError` about duplicate names -- or not at all, if only
        one of them existed. Checking here names the class that is actually wrong.

        Raises
        ------
        TypeError
            If ``discipline_name`` was not declared on the subclass.
        """
        super().__init_subclass__(**kwargs)
        if "discipline_name" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare a 'discipline_name'; it is the key the discipline "
                f"is reported and looked up under."
            )

    def __init__(self) -> None:
        """Create an empty discipline. Parameters are added one at a time by the aircraft.

        There is no guard here against instantiating :class:`Discipline` itself: ``ABCMeta``
        already refuses it, because :attr:`discipline_name` is abstract. An earlier version
        raised its own ``TypeError``, which became unreachable the moment the base grew a real
        abstract member -- and an unreachable line is what this package's coverage gate exists
        to find.
        """
        self._parameters: dict[str, Parameter] = {}

    # -- ownership -----------------------------------------------------------------------

    @classmethod
    def owns(cls, variable: str) -> bool:
        """Return whether ``variable`` belongs to this discipline.

        Parameters
        ----------
        variable : str
            Name of a black-box variable, e.g. ``"ac|geom|wing|S_ref"``.

        Returns
        -------
        bool
            ``True`` if any of :attr:`owned_patterns` matches.
        """
        return any(fnmatchcase(variable, pattern) for pattern in cls.owned_patterns)

    # -- encapsulated state --------------------------------------------------------------

    def add(self, parameter: Parameter) -> None:
        """Take ownership of a parameter.

        Parameters
        ----------
        parameter : Parameter
            The parameter to hold.

        Raises
        ------
        DisciplineError
            If the parameter is not owned by this discipline, or is already held. Both are
            construction errors: the first means the ownership map has a gap, the second means
            the same variable was declared twice in a case file.
        """
        if not self.owns(parameter.name):
            raise DisciplineError(
                f"'{parameter.name}' is not owned by the {self.discipline_name} discipline "
                f"(it owns {list(self.owned_patterns)})."
            )
        if parameter.name in self._parameters:
            raise DisciplineError(f"'{parameter.name}' is already held by the {self.discipline_name} discipline.")
        self._parameters[parameter.name] = parameter

    @property
    def parameters(self) -> Mapping[str, Parameter]:
        """Return the parameters this discipline holds, as a copy.

        A copy rather than the live dictionary: the supported way to change a value is
        :meth:`set`, which validates it, and the supported way to add one is :meth:`add`, which
        checks ownership.
        """
        return dict(self._parameters)

    def parameter(self, name: str) -> Parameter:
        """Return one held parameter.

        Raises
        ------
        KeyError
            If this discipline does not hold it.
        """
        try:
            return self._parameters[name]
        except KeyError:
            raise KeyError(
                f"The {self.discipline_name} discipline does not hold '{name}'. It holds {sorted(self._parameters)}."
            ) from None

    def value(self, name: str) -> float:
        """Return the value of one held parameter, in its own units."""
        return self.parameter(name).value

    def units(self, name: str) -> str | None:
        """Return the units of one held parameter."""
        return self.parameter(name).units

    def set(self, name: str, value: float) -> None:
        """Set the value of one held parameter, in its existing units.

        Raises
        ------
        KeyError
            If this discipline does not hold it. There is no implicit creation: a typo that
            created a new variable would produce a parameter nothing in the box reads.
        """
        self.parameter(name).value = value

    def __contains__(self, name: str) -> bool:
        """Return whether this discipline holds a parameter of that name."""
        return name in self._parameters

    def __iter__(self) -> Iterator[str]:
        """Iterate the names of the held parameters, in insertion order."""
        return iter(self._parameters)

    def __len__(self) -> int:
        """Return the number of held parameters."""
        return len(self._parameters)

    # -- the black-box interface ---------------------------------------------------------

    def apply(self, box: Any) -> None:
        """Write every held parameter into the black box.

        Parameters
        ----------
        box : OpenConceptSizingBox
            Anything with ``set(name, value, units)``.
        """
        for parameter in self._parameters.values():
            box.set(parameter.name, parameter.value, units=parameter.units)

    def collect(self, box: Any) -> dict[str, Any]:
        """Read every reported response back out of the black box.

        Parameters
        ----------
        box : OpenConceptSizingBox
            Anything with ``get(path, units)`` and ``has(path)``.

        Returns
        -------
        dict
            ``response name -> value``. A scalar response is returned as a float, a vector one
            as an array. Optional responses the box does not publish are omitted; required ones
            raise.

        Raises
        ------
        KeyError
            If a required response is absent from this black box. That means the box is not the
            model this discipline was written against, and quietly returning a partial result
            would let a report claim a quantity it never read.
        """
        collected: dict[str, Any] = {}
        for response in self.reported:
            if not box.has(response.path):
                if response.optional:
                    continue
                raise KeyError(
                    f"The {self.discipline_name} discipline reports '{response.name}' from "
                    f"'{response.path}', which this black box does not publish."
                )
            collected[response.name] = box.get(response.path, units=response.units)
        return collected

    def missing(self, box: Any) -> tuple[str, ...]:
        """Return the names of optional responses this black box does not publish."""
        return tuple(r.name for r in self.reported if r.optional and not box.has(r.path))

    def __repr__(self) -> str:
        """Return a representation naming the discipline and how much it holds."""
        return (
            f"{type(self).__name__}(name={self.discipline_name!r}, "
            f"parameters={len(self._parameters)}, responses={len(self.reported)})"
        )
