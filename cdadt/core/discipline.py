"""The discipline abstraction.

A :class:`Discipline` is one engineering domain -- aerodynamics, propulsion, weights,
geometry, stability -- modeled as an object with its own state. It declares what it
provides and what it requires, and delegates its physics to a
:class:`~cdadt.core.provider.Provider`.

Two rules make the abstraction worth having:

**A discipline owns only itself.** :meth:`Discipline.build` adds subsystems to the group it
is given and sets up its own promotions. It never reaches into a sibling discipline, never
inspects the parent, and never mutates anything it did not create. Coupling between
disciplines happens by matching promoted names, and the matching is checked by
:func:`check_coupling` before the model is built.

**No global state.** Everything a discipline needs arrives through its constructor. There
is no registry, no module-level cache, and no import-time side effect. That is what makes
it possible to build several aircraft in one process -- which every sizing loop and every
optimization iteration does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import openmdao.api as om

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

__all__ = ["CouplingError", "Discipline", "check_coupling"]


class CouplingError(RuntimeError):
    """Raised when a set of disciplines cannot be connected.

    Two situations produce it: a discipline requires a variable that nothing in the set
    provides and that the mission does not supply, or two disciplines both claim to
    provide the same variable. Either one would otherwise show up as a silently
    unconnected input sitting at its default value.
    """


class Discipline(ABC):
    """Base class for an engineering discipline.

    Parameters
    ----------
    provider : Provider
        The concrete physics implementation this discipline delegates to.
    config : AircraftConfiguration
        Configuration for the aircraft being modeled. Passed through to the provider and
        available to the discipline for its own bookkeeping.

    Attributes
    ----------
    provider : Provider
        The provider supplied at construction.
    config : AircraftConfiguration
        The configuration supplied at construction.

    Notes
    -----
    Subclasses declare a fixed ``name`` and delegate ``provides``/``requires``/``build`` to
    the provider unless they add coupling of their own. The base class implementations do
    exactly that delegation, so a discipline that is a pure pass-through needs only to
    define :attr:`name`.
    """

    def __init__(self, provider: Provider, config: AircraftConfiguration) -> None:
        self._provider = provider
        self._config = config

    @property
    def provider(self) -> Provider:
        """Return the provider supplying this discipline's physics."""
        return self._provider

    @property
    def config(self) -> AircraftConfiguration:
        """Return the configuration this discipline was constructed with."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the discipline's identifier, e.g. ``"aerodynamics"``.

        Used as the OpenMDAO subsystem name and as the key in reports, so it must be a
        valid OpenMDAO name: no dots, no spaces.
        """

    def provides(self) -> VariableSet:
        """Return the variables this discipline creates as promoted outputs.

        Returns
        -------
        VariableSet
            By default, exactly what the provider provides.
        """
        return self._provider.provides()

    def requires(self) -> VariableSet:
        """Return the variables this discipline consumes as promoted inputs.

        Returns
        -------
        VariableSet
            By default, exactly what the provider requires.
        """
        return self._provider.requires()

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add this discipline's subsystems to ``group``.

        Parameters
        ----------
        group : openmdao.api.Group
            Group to add to. The discipline may add subsystems and set input defaults on
            variables it owns. It must not modify subsystems added by another discipline.
        num_nodes : int
            Number of analysis points in this mission phase.
        flight_phase : str
            Name of the mission phase being built.
        """
        self._provider.build(group, num_nodes=num_nodes, flight_phase=flight_phase)

    def __repr__(self) -> str:
        """Return a representation naming the discipline and its provider."""
        return f"{type(self).__name__}(name={self.name!r}, provider={self._provider.name!r})"


def check_coupling(
    disciplines: Sequence[Discipline],
    externally_supplied: Iterable[Variable] = (),
) -> None:
    """Verify a set of disciplines can be connected before the model is built.

    Two failures are caught here rather than at OpenMDAO setup time, where they would
    appear as an unconnected input quietly holding its declared default:

    * a discipline requires a variable that no other discipline provides and that is not
      supplied externally, and
    * two disciplines both provide the same variable, so which one wins would depend on
      the order they were added.

    Parameters
    ----------
    disciplines : sequence of Discipline
        The disciplines that will be built into one aircraft model.
    externally_supplied : iterable of Variable, optional
        Variables the surrounding model supplies rather than any discipline -- flight
        conditions from the mission phase, throttle, and the like. Declaring them here is
        what distinguishes "supplied by the mission" from "nobody computes this".

    Raises
    ------
    CouplingError
        If any requirement is unsatisfied or any variable is provided more than once. The
        message lists every problem found, not just the first.
    """
    external = VariableSet(externally_supplied)

    provided_by: dict[str, list[str]] = {}
    for discipline in disciplines:
        for variable in discipline.provides():
            provided_by.setdefault(variable.name, []).append(discipline.name)

    duplicates = {name: owners for name, owners in provided_by.items() if len(owners) > 1}

    available = set(provided_by) | set(external.names)
    unsatisfied: dict[str, list[str]] = {}
    for discipline in disciplines:
        for variable in discipline.requires():
            if variable.name not in available:
                unsatisfied.setdefault(variable.name, []).append(discipline.name)

    if not duplicates and not unsatisfied:
        return

    problems = []
    for name, owners in sorted(duplicates.items()):
        problems.append(f"  '{name}' is provided by more than one discipline: {', '.join(sorted(owners))}")
    for name, consumers in sorted(unsatisfied.items()):
        problems.append(
            f"  '{name}' is required by {', '.join(sorted(consumers))} but no discipline provides it "
            f"and it is not declared as externally supplied"
        )
    raise CouplingError(
        "The discipline set cannot be connected:\n" + "\n".join(problems) + "\n"
        "Every promoted input must have exactly one source. Add the missing provider, or "
        "declare the variable in externally_supplied if the mission phase supplies it."
    )
