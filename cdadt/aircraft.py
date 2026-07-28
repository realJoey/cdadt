"""The aircraft definition: design parameters as encapsulated state.

An :class:`AircraftDefinition` holds every ``ac|...`` design parameter for one aircraft, with
its value and its units. It is the single source of the numbers that describe the airframe,
and it is the object the model turns into OpenMDAO independent variables.

The storage format is OpenConcept's own nested-dictionary layout, so the same definition can
be handed to :class:`openconcept.utilities.DictIndepVarComp` without translation::

    {"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}}

A leaf is any mapping with a ``"value"`` key; its structured name is the path to it joined
with pipes, e.g. ``ac|geom|wing|S_ref``. That naming is not cdadt's invention -- it is the
convention every OpenConcept mission phase promotes on, which is why a design parameter set
once at the top of the model reaches all twelve phases.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from openconcept.utilities import DictIndepVarComp

__all__ = ["AircraftDefinition"]


class AircraftDefinition:
    """The design parameters describing one aircraft.

    Parameters
    ----------
    data : mapping
        Nested dictionary in OpenConcept's ``DictIndepVarComp`` layout. It is deep-copied,
        so two definitions built from one literal never share state.

    Raises
    ------
    ValueError
        If the dictionary contains no leaves, or if a leaf's value is not numeric.

    Examples
    --------
    >>> aircraft = AircraftDefinition.from_yaml("cases/b738.yaml")
    >>> aircraft.value("ac|geom|wing|S_ref")
    124.6
    >>> aircraft.units("ac|geom|wing|S_ref")
    'm**2'
    >>> aircraft.set("ac|geom|wing|AR", 11.0)
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = deepcopy(dict(data))
        self._names = tuple(sorted(self._collect(self._data, prefix=())))
        if not self._names:
            raise ValueError(
                "The aircraft definition contains no parameters. A leaf is a mapping with a "
                "'value' key, e.g. {'ac': {'geom': {'wing': {'S_ref': {'value': 124.6}}}}}."
            )
        for name in self._names:
            value = self._leaf(name)["value"]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter '{name}' has a non-numeric value {value!r}.")

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> AircraftDefinition:
        """Load a definition from a YAML file laid out like the nested dictionary.

        Parameters
        ----------
        path : str or pathlib.Path
            File to read.

        Returns
        -------
        AircraftDefinition
            The loaded definition.
        """
        with open(path, encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    # -- inspection ---------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Return every structured parameter name, sorted."""
        return self._names

    @property
    def data(self) -> dict[str, Any]:
        """Return the underlying dictionary.

        Returned by reference so that :class:`~openconcept.utilities.DictIndepVarComp` reads
        the same object this definition mutates; :meth:`set` is the supported way to change a
        value.
        """
        return self._data

    def __contains__(self, name: str) -> bool:
        """Return whether ``name`` is a parameter of this aircraft."""
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        """Iterate the structured parameter names."""
        return iter(self._names)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._names)

    def value(self, name: str) -> float:
        """Return a parameter's value in its stored units.

        Parameters
        ----------
        name : str
            Structured name, e.g. ``"ac|geom|wing|S_ref"``.

        Raises
        ------
        KeyError
            If the parameter is not defined. There is no default: a design parameter that
            nobody chose is not a design parameter.
        """
        return float(self._leaf(name)["value"])

    def units(self, name: str) -> str | None:
        """Return a parameter's units, or ``None`` if it is dimensionless."""
        return self._leaf(name).get("units")

    def set(self, name: str, value: float) -> None:
        """Set a parameter's value, in its existing units.

        Parameters
        ----------
        name : str
            Structured name of an already-defined parameter.
        value : float
            New value.

        Raises
        ------
        KeyError
            If the parameter is not already defined. Adding parameters through this method
            would let a typo create a variable nothing reads.
        """
        self._leaf(name)["value"] = float(value)

    # -- model building -----------------------------------------------------------------

    def component(self, names: tuple[str, ...] | None = None) -> DictIndepVarComp:
        """Return an OpenConcept ``DictIndepVarComp`` publishing these parameters.

        Every output is an independent variable, which is what makes any of them addressable
        as an optimizer design variable.

        Parameters
        ----------
        names : tuple of str, optional
            Parameters to publish. Defaults to all of them.

        Returns
        -------
        openconcept.utilities.DictIndepVarComp
            Component with one output per requested parameter, named and united as stored.
        """
        component = DictIndepVarComp(self._data)
        for name in names if names is not None else self._names:
            component.add_output_from_dict(name)
        return component

    def as_array(self, name: str) -> np.ndarray:
        """Return a parameter's value shaped as OpenMDAO stores it: a length-one array."""
        return np.array([self.value(name)])

    # -- internals ----------------------------------------------------------------------

    @staticmethod
    def _collect(node: Mapping[str, Any], prefix: tuple[str, ...]) -> list[str]:
        """Return the structured names of every leaf at or below ``node``."""
        if "value" in node:
            return ["|".join(prefix)]
        names: list[str] = []
        for key, child in node.items():
            if isinstance(child, Mapping):
                names += AircraftDefinition._collect(child, (*prefix, key))
        return names

    def _leaf(self, name: str) -> dict[str, Any]:
        """Return the leaf dictionary for a structured name."""
        node: Any = self._data
        for part in name.split("|"):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(f"'{name}' is not defined in this aircraft. Defined parameters: {list(self._names)}")
            node = node[part]
        if not isinstance(node, Mapping) or "value" not in node:
            raise KeyError(f"'{name}' names a group of parameters, not a parameter.")
        return node

    def __repr__(self) -> str:
        """Return a representation naming the parameter count."""
        return f"AircraftDefinition({len(self._names)} parameters)"
