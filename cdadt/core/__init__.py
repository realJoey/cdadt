"""Core abstractions shared by every part of cdadt.

This package defines the vocabulary the rest of the framework is written in:

:mod:`cdadt.core.variables`
    :class:`~cdadt.core.variables.Variable` and
    :class:`~cdadt.core.variables.VariableSet` -- typed, unit-validated declarations of
    what a piece of the model consumes and produces.

:mod:`cdadt.core.configuration`
    :class:`~cdadt.core.configuration.AircraftConfiguration` -- the immutable design state,
    with no facility for reading a value with a fallback.

:mod:`cdadt.core.provider`
    :class:`~cdadt.core.provider.Provider` -- the strategy interface for a concrete physics
    implementation.

:mod:`cdadt.core.discipline`
    :class:`~cdadt.core.discipline.Discipline` -- one engineering domain, delegating its
    physics to a provider, plus :func:`~cdadt.core.discipline.check_coupling` for verifying
    a discipline set connects before anything is built.

Nothing in this package holds mutable module-level state.
"""

from cdadt.core.configuration import (
    AircraftConfiguration,
    ConfigurationError,
    MissingConfigurationError,
)
from cdadt.core.discipline import (
    CouplingError,
    Discipline,
    DisciplineGroup,
    DisciplineScope,
    check_coupling,
    resolve_input_units,
)
from cdadt.core.provider import Provider, ProviderContractError
from cdadt.core.variables import Variable, VariableConflictError, VariableSet

__all__ = [
    "AircraftConfiguration",
    "ConfigurationError",
    "CouplingError",
    "Discipline",
    "DisciplineGroup",
    "DisciplineScope",
    "MissingConfigurationError",
    "Provider",
    "ProviderContractError",
    "Variable",
    "VariableConflictError",
    "VariableSet",
    "check_coupling",
    "resolve_input_units",
]
