"""The provider strategy: where a discipline's physics actually comes from.

A :class:`~cdadt.core.discipline.Discipline` says *what* it computes. A :class:`Provider`
says *how*. Separating the two is what lets an empirical drag buildup and a vortex-lattice
analysis both satisfy "aerodynamics" without the surrounding model knowing which is in use.

A provider is responsible for one thing: adding OpenMDAO subsystems to a group so that,
afterwards, every variable the provider declares in :meth:`Provider.provides` is available
as a promoted output of that group. It reads its method constants from the
:class:`~cdadt.core.configuration.AircraftConfiguration` it is given, which is why
providers hold no numeric literals of their own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import openmdao.api as om

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.variables import VariableSet

__all__ = ["Provider", "ProviderContractError"]


class ProviderContractError(RuntimeError):
    """Raised when a provider does not deliver what it declared.

    Declaring a variable in :meth:`Provider.provides` and then failing to create it is the
    failure mode this catches. It is checked after the model is built, so it names the
    provider and the variables that are missing rather than surfacing later as an
    OpenMDAO connection error in an unrelated part of the tree.
    """


class Provider(ABC):
    """Base class for a concrete physics implementation behind a discipline.

    Subclasses declare the variables they consume and produce, and build the OpenMDAO
    subsystems that produce them. All state lives on the instance; providers must not read
    or write module-level state.

    Parameters
    ----------
    config : AircraftConfiguration
        The configuration this provider reads its method constants from. Providers should
        call :meth:`~cdadt.core.configuration.AircraftConfiguration.require_all` in
        :meth:`validate_configuration` so that a missing constant is reported before the
        model is built rather than during a solver iteration.

    Attributes
    ----------
    config : AircraftConfiguration
        The configuration passed at construction. Read-only by convention; the
        configuration object is itself immutable.
    """

    def __init__(self, config: AircraftConfiguration) -> None:
        self._config = config
        self.validate_configuration()

    @property
    def config(self) -> AircraftConfiguration:
        """Return the configuration this provider reads from."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short identifier for this provider, used in subsystem names and reports."""

    @property
    @abstractmethod
    def reference(self) -> str:
        """Return a citation for the method this provider implements.

        This appears in the run report next to every result the provider influenced. A
        provider that wraps a third-party component cites that component; a provider that
        implements a published method cites the publication and equation numbers.
        """

    @abstractmethod
    def provides(self) -> VariableSet:
        """Return the variables this provider guarantees to create as promoted outputs."""

    @abstractmethod
    def requires(self) -> VariableSet:
        """Return the variables this provider consumes as promoted inputs.

        These must be supplied by another provider, by the mission phase the discipline
        lives in, or by the configuration.
        """

    @abstractmethod
    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add this provider's subsystems to ``group``.

        Parameters
        ----------
        group : openmdao.api.Group
            The group to add subsystems to. The provider adds subsystems and sets up
            promotions; it must not modify subsystems it did not add, and must not reach
            outside ``group``.
        num_nodes : int
            Number of analysis points in the mission phase this instance serves. Vectorized
            variables have this length.
        flight_phase : str
            Name of the mission phase, e.g. ``"cruise"`` or ``"v0v1"``. Providers whose
            physics changes with configuration (flaps down for takeoff, gear extended for
            landing) branch on this.
        """

    def validate_configuration(self) -> None:  # noqa: B027 -- optional hook, deliberately not abstract
        """Check that every configuration entry this provider needs is present.

        The default implementation accepts any configuration. Subclasses that read method
        constants override this and call
        :meth:`~cdadt.core.configuration.AircraftConfiguration.require_all`, so that a
        configuration missing a constant fails at construction with a list of what is
        absent, rather than at solve time with a fallback value nobody chose.
        """

    def __repr__(self) -> str:
        """Return a representation naming the provider class and its identifier."""
        return f"{type(self).__name__}(name={self.name!r})"
