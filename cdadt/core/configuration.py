"""The design state of an aircraft, loaded from configuration and read by disciplines.

An :class:`AircraftConfiguration` holds every number that describes a design and every
number a method needs in order to run. It deliberately provides **no way to read a value
with a fallback**. Requesting a name that is not configured raises. That is a structural
rule, not a stylistic one: a "reference value" supplied as a default is a hardcoded
constant that happens to have a citation, and once a method can fall back to one, a
configuration that never set it produces a plausible answer that nobody chose.

The storage format matches OpenConcept's ``DictIndepVarComp`` convention so that a
configuration can feed an OpenConcept model directly: a nested dictionary whose leaves are
``{"value": ..., "units": ...}`` mappings, addressed by a pipe-separated path such as
``ac|geom|wing|S_ref``.
"""

from __future__ import annotations

import copy
import numbers
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from openmdao.utils.units import convert_units

from cdadt.core.variables import is_valid_units

__all__ = ["AircraftConfiguration", "ConfigurationError", "MissingConfigurationError"]

SEPARATOR = "|"
VALUE_KEY = "value"
UNITS_KEY = "units"
SOURCE_KEY = "source"


class ConfigurationError(ValueError):
    """Raised when a configuration tree is malformed."""


class MissingConfigurationError(KeyError):
    """Raised when a required configuration entry is absent.

    This is the error that enforces cdadt's no-defaults rule. It is a distinct type so that
    tests can assert a method refuses to run without its configured constants, rather than
    asserting on a message.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"'{name}' is not present in the configuration and has no default. "
            f"cdadt does not supply fallback values -- add it to the configuration file. "
            f"Nearest configured names: {_nearest_names(name, available)}"
        )

    @classmethod
    def for_missing_set(cls, missing: list[str], available: list[str]) -> MissingConfigurationError:
        """Build an error reporting several absent entries at once.

        Parameters
        ----------
        missing : list of str
            Every name that was required and not found.
        available : list of str
            Every configured name, used to suggest near matches.

        Returns
        -------
        MissingConfigurationError
            An error whose ``name`` attribute is the first missing entry and whose
            ``missing`` attribute lists them all.
        """
        error = cls(missing[0], available)
        error.missing = list(missing)
        error.args = (
            f"{len(missing)} required configuration entries are absent and have no defaults: "
            f"{', '.join(missing)}. cdadt does not supply fallback values -- add them to the "
            f"configuration file.",
        )
        return error

    def __str__(self) -> str:
        """Return the message without ``KeyError``'s surrounding quotes."""
        return self.args[0]


def _nearest_names(name: str, available: list[str], limit: int = 5) -> list[str]:
    """Return configured names sharing the longest leading path with ``name``.

    Parameters
    ----------
    name : str
        The name that was requested and not found.
    available : list of str
        All configured names.
    limit : int, optional
        Maximum number of suggestions to return. Default 5.

    Returns
    -------
    list of str
        Suggestions ordered by how much of the pipe-separated path they share.
    """
    requested = name.split(SEPARATOR)

    def shared_depth(candidate: str) -> int:
        parts = candidate.split(SEPARATOR)
        depth = 0
        for a, b in zip(requested, parts, strict=False):
            if a != b:
                break
            depth += 1
        return depth

    ranked = sorted(available, key=lambda c: (-shared_depth(c), c))
    return [c for c in ranked[:limit] if shared_depth(c) > 0] or sorted(available)[:limit]


class AircraftConfiguration:
    """Immutable design and method configuration for one aircraft.

    Parameters
    ----------
    data : mapping
        Nested dictionary whose leaves are ``{"value": ..., "units": ...}`` mappings.
        ``units`` may be omitted or ``None`` for dimensionless quantities. An optional
        ``source`` string may be attached to any leaf to record where the number came
        from; :meth:`source` reads it back for the traceability report.

    Raises
    ------
    ConfigurationError
        If any leaf is malformed or declares an invalid unit string.

    Examples
    --------
    >>> config = AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}})
    >>> config.value("ac|geom|wing|S_ref", units="ft**2")
    array([1341.1854...])
    """

    __slots__ = ("_data", "_names")

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"Configuration data must be a mapping, got {type(data).__name__}")
        validated = copy.deepcopy(dict(data))
        names = sorted(_walk_leaves(validated, prefix=()))
        object.__setattr__(self, "_data", validated)
        object.__setattr__(self, "_names", tuple(names))

    def __setattr__(self, name: str, value: object) -> None:
        """Reject attribute assignment; a configuration is a fixed record of a design."""
        raise AttributeError(f"AircraftConfiguration is immutable; cannot set {name!r}. Use with_overrides().")

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> AircraftConfiguration:
        """Load a configuration from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a YAML file in the nested ``{"value": ..., "units": ...}`` format.

        Returns
        -------
        AircraftConfiguration
            The loaded configuration.

        Raises
        ------
        ConfigurationError
            If the file does not contain a mapping, or any leaf is malformed.
        """
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, Mapping):
            raise ConfigurationError(f"{path} must contain a mapping at the top level, got {type(loaded).__name__}")
        return cls(loaded)

    def with_overrides(self, overrides: Mapping[str, Any]) -> AircraftConfiguration:
        """Return a new configuration with some entries replaced.

        Parameters
        ----------
        overrides : mapping
            Flat mapping of pipe-separated name to either a raw value (units are inherited
            from the existing entry) or a full ``{"value": ..., "units": ...}`` leaf.

        Returns
        -------
        AircraftConfiguration
            A new configuration. The receiver is unchanged.

        Raises
        ------
        MissingConfigurationError
            If a raw value is supplied for a name that is not already configured, since
            there would be no units to inherit.
        """
        data = copy.deepcopy(self._data)
        for name, override in overrides.items():
            parts = name.split(SEPARATOR)
            node: Any = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            if isinstance(override, Mapping) and VALUE_KEY in override:
                leaf = dict(override)
            else:
                if name not in self._names:
                    raise MissingConfigurationError(name, list(self._names))
                leaf = dict(self._leaf(name))
                leaf[VALUE_KEY] = override
            node[parts[-1]] = leaf
        return AircraftConfiguration(data)

    # -- reading ------------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Return every configured name, as pipe-separated paths, in sorted order."""
        return self._names

    def __contains__(self, name: str) -> bool:
        """Return whether ``name`` is configured."""
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        """Iterate configured names."""
        return iter(self._names)

    def __len__(self) -> int:
        """Return the number of configured entries."""
        return len(self._names)

    def value(self, name: str, units: str | None = None) -> np.ndarray:
        """Return a configured value, optionally converted to requested units.

        There is no ``default`` parameter, and there will not be one. A method that needs a
        constant must have that constant configured.

        Parameters
        ----------
        name : str
            Pipe-separated path, e.g. ``"ac|geom|wing|S_ref"``.
        units : str or None, optional
            Units to convert to. If ``None``, the value is returned in its configured
            units.

        Returns
        -------
        numpy.ndarray
            The value, always as an array so that scalars and vectors are handled
            uniformly by callers.

        Raises
        ------
        MissingConfigurationError
            If ``name`` is not configured.
        ConfigurationError
            If ``units`` is requested for a dimensionless entry, or if the configured units
            are not convertible to ``units``.
        """
        leaf = self._leaf(name)
        raw = leaf[VALUE_KEY]
        value = np.array([raw], dtype=float) if isinstance(raw, numbers.Number) else np.asarray(raw, dtype=float)

        configured_units = leaf.get(UNITS_KEY)
        if units is None or units == configured_units:
            return value
        if configured_units is None:
            raise ConfigurationError(f"'{name}' is configured as dimensionless but was requested in units '{units}'.")
        try:
            return np.asarray(convert_units(value, configured_units, units), dtype=float)
        except Exception as err:
            raise ConfigurationError(
                f"Cannot convert '{name}' from its configured units '{configured_units}' to '{units}'."
            ) from err

    def scalar(self, name: str, units: str | None = None) -> float:
        """Return a configured scalar value.

        Parameters
        ----------
        name : str
            Pipe-separated path.
        units : str or None, optional
            Units to convert to.

        Returns
        -------
        float
            The single configured value.

        Raises
        ------
        MissingConfigurationError
            If ``name`` is not configured.
        ConfigurationError
            If the entry holds more than one element.
        """
        value = self.value(name, units=units)
        if value.size != 1:
            raise ConfigurationError(f"'{name}' holds {value.size} elements; scalar() requires exactly one.")
        return float(value.reshape(-1)[0])

    def units(self, name: str) -> str | None:
        """Return the configured units of an entry, or ``None`` if dimensionless.

        Raises
        ------
        MissingConfigurationError
            If ``name`` is not configured.
        """
        return self._leaf(name).get(UNITS_KEY)

    def source(self, name: str) -> str | None:
        """Return the recorded provenance of an entry, or ``None`` if none was recorded.

        Provenance strings are what the certification traceability report cites for every
        method constant, so that a reader can check a number against its reference.

        Raises
        ------
        MissingConfigurationError
            If ``name`` is not configured.
        """
        return self._leaf(name).get(SOURCE_KEY)

    def require_all(self, names: list[str]) -> None:
        """Assert that every name in ``names`` is configured, reporting all that are not.

        Reporting every missing name at once matters when a discipline needs a dozen
        constants: fixing them one traceback at a time is how a configuration ends up with
        a value that was guessed to make an error go away.

        Parameters
        ----------
        names : list of str
            Pipe-separated paths that must all be present.

        Raises
        ------
        MissingConfigurationError
            If any name is absent. The message lists every absent name.
        """
        missing = [name for name in names if name not in self._names]
        if missing:
            raise MissingConfigurationError.for_missing_set(missing, list(self._names))

    def as_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying nested dictionary.

        The copy is what makes this safe to hand to OpenConcept's ``DictIndepVarComp``:
        mutations by the consumer cannot reach back into the configuration.
        """
        return copy.deepcopy(self._data)

    def subtree(self, prefix: str) -> AircraftConfiguration:
        """Return the configuration rooted at ``prefix``, still addressed by full names.

        Disciplines receive the whole configuration but are documented in terms of the
        subtree they own; this makes that subtree explicit for tests and reporting.

        Parameters
        ----------
        prefix : str
            Pipe-separated path prefix, e.g. ``"ac|geom"``.

        Returns
        -------
        AircraftConfiguration
            A configuration containing only entries under ``prefix``, with their original
            full names preserved.

        Raises
        ------
        MissingConfigurationError
            If no configured name lies under ``prefix``.
        """
        matching = [name for name in self._names if name == prefix or name.startswith(prefix + SEPARATOR)]
        if not matching:
            raise MissingConfigurationError(prefix, list(self._names))

        data: dict[str, Any] = {}
        for name in matching:
            parts = name.split(SEPARATOR)
            node: Any = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = copy.deepcopy(self._leaf(name))
        return AircraftConfiguration(data)

    def __repr__(self) -> str:
        """Return a representation reporting how many entries are configured."""
        return f"AircraftConfiguration({len(self._names)} entries)"

    # -- internals ----------------------------------------------------------------------

    def _leaf(self, name: str) -> Mapping[str, Any]:
        """Return the raw leaf mapping for ``name``.

        Raises
        ------
        MissingConfigurationError
            If ``name`` is not configured.
        """
        node: Any = self._data
        for part in name.split(SEPARATOR):
            if not isinstance(node, Mapping) or part not in node:
                raise MissingConfigurationError(name, list(self._names))
            node = node[part]
        if not _is_leaf(node):
            raise MissingConfigurationError(name, list(self._names))
        return node


def _is_leaf(node: Any) -> bool:
    """Return whether a node is a value leaf rather than an interior branch."""
    return isinstance(node, Mapping) and VALUE_KEY in node


def _walk_leaves(node: Any, prefix: tuple[str, ...]) -> Iterator[str]:
    """Yield the pipe-separated name of every value leaf under ``node``.

    Validates each leaf as it goes: a leaf must carry a numeric or array ``value`` and, if
    it declares ``units``, that unit string must be one OpenMDAO recognizes.

    Raises
    ------
    ConfigurationError
        If a leaf is malformed or declares an invalid unit string.
    """
    if not isinstance(node, Mapping):
        raise ConfigurationError(
            f"'{SEPARATOR.join(prefix)}' must be a mapping or a {{'value': ...}} leaf, got {type(node).__name__}"
        )

    if _is_leaf(node):
        name = SEPARATOR.join(prefix)
        raw = node[VALUE_KEY]
        if not isinstance(raw, (numbers.Number, list, tuple, np.ndarray)):
            raise ConfigurationError(f"'{name}' has a non-numeric value of type {type(raw).__name__}")
        declared_units = node.get(UNITS_KEY)
        if not is_valid_units(declared_units):
            raise ConfigurationError(f"'{name}' declares invalid units '{declared_units}'")
        unexpected = set(node) - {VALUE_KEY, UNITS_KEY, SOURCE_KEY}
        if unexpected:
            raise ConfigurationError(
                f"'{name}' has unexpected keys {sorted(unexpected)}; "
                f"a leaf may contain only {VALUE_KEY!r}, {UNITS_KEY!r} and {SOURCE_KEY!r}"
            )
        yield name
        return

    if not node:
        raise ConfigurationError(f"'{SEPARATOR.join(prefix)}' is an empty branch with no configured values")

    for key, child in node.items():
        yield from _walk_leaves(child, (*prefix, str(key)))
