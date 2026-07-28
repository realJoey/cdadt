"""The aircraft: the disciplines that describe an airframe, and the routing between them.

An :class:`Aircraft` is a composition of discipline objects, not a bag of numbers. Given the
parameters a case file declares, it hands each one to the discipline that owns it and refuses
any that no discipline owns or that two would claim. What comes out is an object where
``aircraft["geometry"].value("ac|geom|wing|S_ref")`` is the question you ask, rather than a flat
dictionary where a wing area and a cabin pressure sit side by side.

Routing is the point. It is what makes the ownership map testable: if the black box publishes a
settable variable that no discipline claims, building an aircraft that sets it fails by name,
and a contract test walks the box's whole settable set to prove that never happens silently.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from cdadt.disciplines import AIRCRAFT_DISCIPLINES, Discipline
from cdadt.parameters import Parameter

__all__ = ["Aircraft", "AircraftError"]


class AircraftError(Exception):
    """Raised when a parameter cannot be routed to exactly one discipline."""


class Aircraft:
    """One airframe, described by its disciplines.

    Parameters
    ----------
    parameters : sequence of Parameter
        The design parameters to route. Every one must be owned by exactly one discipline.
    disciplines : sequence of type, optional
        Discipline classes to compose. Default
        :data:`~cdadt.disciplines.AIRCRAFT_DISCIPLINES`. Passing a different set is how a study
        adds a domain: subclass :class:`~cdadt.disciplines.base.Discipline`, declare what it
        owns and reports, and include it here.

    Raises
    ------
    AircraftError
        If a parameter is owned by no discipline or by more than one, or if two disciplines
        share a name.

    Examples
    --------
    >>> aircraft = Aircraft([Parameter("ac|geom|wing|AR", 9.45)])
    >>> aircraft["geometry"].value("ac|geom|wing|AR")
    9.45
    >>> aircraft.owner("ac|geom|wing|AR").discipline_name
    'geometry'
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        disciplines: Sequence[type[Discipline]] = AIRCRAFT_DISCIPLINES,
    ) -> None:
        self._disciplines: dict[str, Discipline] = {}
        for discipline_class in disciplines:
            name = discipline_class.discipline_name
            if name in self._disciplines:
                raise AircraftError(f"Two disciplines are both called '{name}'.")
            self._disciplines[name] = discipline_class()

        for parameter in parameters:
            self._route(parameter)

    # -- routing -------------------------------------------------------------------------

    def _route(self, parameter: Parameter) -> None:
        """Give ``parameter`` to the one discipline that owns it."""
        owners = [d for d in self._disciplines.values() if d.owns(parameter.name)]
        if not owners:
            raise AircraftError(
                f"No discipline owns '{parameter.name}'. Either it is misspelled, or the ownership map "
                f"needs extending: the disciplines present are {sorted(self._disciplines)}."
            )
        if len(owners) > 1:
            raise AircraftError(
                f"'{parameter.name}' is claimed by {[d.discipline_name for d in owners]}. "
                f"Ownership must be disjoint, or the same variable would be set twice."
            )
        owners[0].add(parameter)

    def owner(self, variable: str) -> Discipline:
        """Return the discipline that owns ``variable``.

        Raises
        ------
        AircraftError
            If none does, or more than one does.
        """
        owners = [d for d in self._disciplines.values() if d.owns(variable)]
        if len(owners) != 1:
            raise AircraftError(f"'{variable}' is owned by {len(owners)} disciplines; exactly one is required.")
        return owners[0]

    def unowned(self, variables: Sequence[str]) -> tuple[str, ...]:
        """Return the members of ``variables`` that no discipline owns.

        Used by the contract test that walks every settable variable of the black box: an input
        cdadt can set but no discipline claims is a gap in the model of the interface, whether
        or not any case file happens to set it.
        """
        return tuple(v for v in variables if not any(d.owns(v) for d in self._disciplines.values()))

    # -- state ---------------------------------------------------------------------------

    @property
    def disciplines(self) -> Mapping[str, Discipline]:
        """Return the disciplines, keyed by name, in composition order."""
        return dict(self._disciplines)

    def __getitem__(self, name: str) -> Discipline:
        """Return one discipline by name."""
        try:
            return self._disciplines[name]
        except KeyError:
            raise KeyError(f"This aircraft has no '{name}' discipline; it has {sorted(self._disciplines)}.") from None

    def __contains__(self, name: str) -> bool:
        """Return whether a discipline of that name is present."""
        return name in self._disciplines

    def __iter__(self) -> Iterator[str]:
        """Iterate the discipline names."""
        return iter(self._disciplines)

    def __len__(self) -> int:
        """Return the number of disciplines."""
        return len(self._disciplines)

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Return every held parameter, flattened, keyed by variable name."""
        flat: dict[str, Parameter] = {}
        for discipline in self._disciplines.values():
            flat.update(discipline.parameters)
        return flat

    def value(self, variable: str) -> float:
        """Return the value of one parameter, wherever it lives."""
        return self.owner(variable).value(variable)

    def set(self, variable: str, value: float) -> None:
        """Set the value of one parameter, in its existing units."""
        self.owner(variable).set(variable, value)

    # -- the black-box interface ---------------------------------------------------------

    def apply(self, box: Any) -> None:
        """Write every parameter of every discipline into the black box."""
        for discipline in self._disciplines.values():
            discipline.apply(box)

    def collect(self, box: Any) -> dict[str, dict[str, Any]]:
        """Read every discipline's responses back out of the black box.

        Returns
        -------
        dict
            ``discipline name -> {response name -> value}``.
        """
        return {name: discipline.collect(box) for name, discipline in self._disciplines.items()}

    def missing(self, box: Any) -> dict[str, tuple[str, ...]]:
        """Return, per discipline, the optional responses this black box does not publish."""
        return {
            name: discipline.missing(box) for name, discipline in self._disciplines.items() if discipline.missing(box)
        }

    def __repr__(self) -> str:
        """Return a representation naming the discipline and parameter counts."""
        return f"Aircraft({len(self._disciplines)} disciplines, {len(self.parameters)} parameters)"
