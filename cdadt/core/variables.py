"""Typed variable declarations and immutable variable collections.

Disciplines in cdadt declare what they provide and what they require. Those declarations
are the basis of every contract check in the test suite, so they are objects with validated
units rather than bare strings.

Names follow the OpenConcept convention: a pipe-separated path such as
``ac|geom|wing|S_ref`` for a design parameter or ``fltcond|CL`` for a flight condition.
Using the same convention means a cdadt :class:`VariableSet` can be checked directly against
the promoted names of a built OpenConcept model.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from openmdao.utils.units import valid_units

__all__ = ["Variable", "VariableConflictError", "VariableSet", "is_valid_units"]


def is_valid_units(units: str | None) -> bool:
    """Return whether ``units`` is a unit string OpenMDAO can interpret.

    Parameters
    ----------
    units : str or None
        Candidate unit string. ``None`` means dimensionless and is always valid.

    Returns
    -------
    bool
        ``True`` if OpenMDAO recognizes the string.

    Notes
    -----
    OpenMDAO's own ``valid_units`` evaluates the unit string as a Python expression and so
    raises ``SyntaxError`` on malformed input such as ``"m**"`` or ``"kg/"`` rather than
    returning ``False``. This wrapper converts every failure mode into a boolean, so
    callers can report a bad unit string as a validation error rather than propagating a
    ``SyntaxError`` from deep inside the units library.
    """
    if units is None:
        return True
    if not isinstance(units, str):
        return False
    try:
        return bool(valid_units(units))
    except Exception:
        return False


class VariableConflictError(ValueError):
    """Raised when two incompatible declarations of the same variable name are combined.

    Two declarations conflict when they share a name but disagree on units or on whether
    the variable is vectorized over the nodes of a mission phase. Silently keeping one of
    them would let a unit error survive into the model, so combining them is an error.
    """


@dataclass(frozen=True)
class Variable:
    """A single declared model variable.

    Instances are immutable and hashable, so a :class:`VariableSet` can be treated as a
    value rather than as mutable shared state.

    Parameters
    ----------
    name : str
        Pipe-separated variable name, e.g. ``"ac|geom|wing|S_ref"`` or ``"fltcond|CL"``.
    units : str or None
        OpenMDAO-compatible unit string, or ``None`` for a dimensionless variable.
        Validated against OpenMDAO's unit library on construction.
    vectorized : bool, optional
        ``True`` if the variable carries one value per analysis node within a mission
        phase; ``False`` if it is a single scalar for the whole aircraft. Default
        ``False``.
    description : str, optional
        Human-readable description. Carried into documentation and error messages.

    Raises
    ------
    ValueError
        If ``name`` is empty or ``units`` is not a valid OpenMDAO unit string.

    Examples
    --------
    >>> Variable("ac|geom|wing|S_ref", "m**2", description="Wing reference area")
    Variable(name='ac|geom|wing|S_ref', units='m**2', vectorized=False, ...)
    """

    name: str
    units: str | None
    vectorized: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Variable name must be a non-empty string")
        if not is_valid_units(self.units):
            raise ValueError(
                f"'{self.units}' is not a valid OpenMDAO unit string (declared for variable "
                f"'{self.name}'). Use None for a dimensionless variable."
            )

    def conflicts_with(self, other: Variable) -> bool:
        """Return whether two same-named declarations are mutually inconsistent.

        Parameters
        ----------
        other : Variable
            The other declaration to compare against.

        Returns
        -------
        bool
            ``True`` if the two share a name but disagree on units or vectorization.
            Descriptions may differ without conflict.
        """
        if self.name != other.name:
            return False
        return self.units != other.units or self.vectorized != other.vectorized


class VariableSet:
    """An immutable, name-keyed collection of :class:`Variable` declarations.

    Set operations return new instances; no method mutates the receiver. Union raises on
    conflicting declarations of the same name rather than picking a winner, so a units
    disagreement between two disciplines surfaces at assembly time.

    Parameters
    ----------
    variables : iterable of Variable, optional
        Declarations to include. Duplicate names are permitted only when the declarations
        are identical in units and vectorization.

    Raises
    ------
    VariableConflictError
        If ``variables`` contains conflicting declarations of the same name.
    """

    __slots__ = ("_by_name",)

    def __init__(self, variables: Iterable[Variable] = ()) -> None:
        by_name: dict[str, Variable] = {}
        for variable in variables:
            existing = by_name.get(variable.name)
            if existing is not None and existing.conflicts_with(variable):
                raise VariableConflictError(
                    f"Conflicting declarations of '{variable.name}': "
                    f"(units={existing.units!r}, vectorized={existing.vectorized}) vs "
                    f"(units={variable.units!r}, vectorized={variable.vectorized})"
                )
            by_name.setdefault(variable.name, variable)
        object.__setattr__(self, "_by_name", dict(by_name))

    def __setattr__(self, name: str, value: object) -> None:
        """Reject attribute assignment; a :class:`VariableSet` is a value, not shared state."""
        raise AttributeError(
            f"VariableSet is immutable; cannot set {name!r}. Build a new set with union/difference instead."
        )

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion; a :class:`VariableSet` is immutable."""
        raise AttributeError(f"VariableSet is immutable; cannot delete {name!r}")

    # -- container protocol -------------------------------------------------------------

    def __iter__(self) -> Iterator[Variable]:
        """Iterate declarations in insertion order."""
        return iter(self._by_name.values())

    def __len__(self) -> int:
        """Return the number of declared variables."""
        return len(self._by_name)

    def __contains__(self, item: object) -> bool:
        """Return whether a name (``str``) or a declaration (:class:`Variable`) is present.

        A :class:`Variable` is present only if an identical declaration is present; a
        same-named declaration with different units is *not* considered present.
        """
        if isinstance(item, str):
            return item in self._by_name
        if isinstance(item, Variable):
            return self._by_name.get(item.name) == item
        return False

    def __getitem__(self, name: str) -> Variable:
        """Return the declaration for ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not declared. The message lists the declared names, because a
            missing variable is almost always a typo in a pipe-separated path.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"'{name}' is not declared in this VariableSet. Declared: {sorted(self._by_name)}") from None

    # -- set algebra --------------------------------------------------------------------

    @property
    def names(self) -> frozenset[str]:
        """Return the declared names as a frozen set."""
        return frozenset(self._by_name)

    def union(self, other: VariableSet) -> VariableSet:
        """Return the union of two sets.

        Parameters
        ----------
        other : VariableSet
            Set to merge in.

        Returns
        -------
        VariableSet
            All declarations from both sets.

        Raises
        ------
        VariableConflictError
            If the two sets declare the same name with different units or vectorization.
        """
        return VariableSet([*self, *other])

    def intersection(self, other: VariableSet) -> VariableSet:
        """Return declarations from this set whose names also appear in ``other``."""
        return VariableSet(v for v in self if v.name in other.names)

    def difference(self, other: VariableSet) -> VariableSet:
        """Return declarations from this set whose names do not appear in ``other``."""
        return VariableSet(v for v in self if v.name not in other.names)

    def __or__(self, other: VariableSet) -> VariableSet:
        """Alias for :meth:`union`."""
        return self.union(other)

    def __and__(self, other: VariableSet) -> VariableSet:
        """Alias for :meth:`intersection`."""
        return self.intersection(other)

    def __sub__(self, other: VariableSet) -> VariableSet:
        """Alias for :meth:`difference`."""
        return self.difference(other)

    def __eq__(self, other: object) -> bool:
        """Return whether two sets hold exactly the same declarations."""
        if not isinstance(other, VariableSet):
            return NotImplemented
        return self._by_name == other._by_name

    def __hash__(self) -> int:
        """Hash on the frozen set of declarations, so a :class:`VariableSet` is usable as a key."""
        return hash(frozenset(self._by_name.values()))

    def __repr__(self) -> str:
        """Return a representation listing the declared names."""
        return f"VariableSet({sorted(self._by_name)})"
