"""The two value objects every discipline is built from.

A :class:`Parameter` is one number cdadt *puts into* the black box: a value, the units it is
expressed in, and where it came from. A :class:`Response` is one quantity cdadt *reads back
out*: a name, the path it lives at inside the box, and the units to read it in.

Both are deliberately small and both encapsulate their state. A parameter's value is reachable
only through a property that validates it, so a discipline cannot end up holding a string, a
``None`` or a ``NaN`` and only discover it when OpenMDAO raises three layers down. Neither
class knows anything about OpenMDAO or OpenConcept; they are plain state.

Provenance is not optional on a parameter. A number with no recorded source is
indistinguishable from a guess in the report that quotes it, and this tool exists to produce
reports that are argued from.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["Parameter", "Response"]


class Parameter:
    """One design parameter set on the black box.

    Parameters
    ----------
    name : str
        Name of the variable as the black box publishes it, e.g. ``"ac|geom|wing|S_ref"``.
    value : float
        Numeric value, expressed in ``units``.
    units : str or None, optional
        Units of ``value``, in OpenMDAO's unit syntax (``"m**2"``, ``"lbf"``, ``"kn"``).
        ``None`` for a dimensionless quantity such as aspect ratio.
    source : str, optional
        Where the number came from -- a reference, a measurement, a requirement, a decision.
        Default ``""``, which is allowed but is reported as ``unsourced``.

    Raises
    ------
    ValueError
        If ``name`` is empty, or if ``value`` is not a finite real number.

    Examples
    --------
    >>> area = Parameter("ac|geom|wing|S_ref", 124.6, "m**2", source="b737.org.uk tech specs")
    >>> area.value
    124.6
    >>> area.value = 130.0
    >>> area
    Parameter('ac|geom|wing|S_ref', 130.0, 'm**2')
    """

    __slots__ = ("_name", "_source", "_units", "_value")

    def __init__(self, name: str, value: float, units: str | None = None, source: str = "") -> None:
        if not name:
            raise ValueError("A parameter must have a name.")
        self._name = str(name)
        self._units = units
        self._source = str(source)
        self._value = 0.0
        self.value = value  # validated by the setter

    # -- state ---------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of the variable in the black box. Fixed for the life of the parameter."""
        return self._name

    @property
    def units(self) -> str | None:
        """Units the value is expressed in, or ``None`` if dimensionless. Fixed."""
        return self._units

    @property
    def source(self) -> str:
        """Where the value came from. Fixed."""
        return self._source

    @property
    def value(self) -> float:
        """The numeric value, in :attr:`units`.

        This is the only mutable state in the class, and it is mutable because an optimizer
        changing a design variable is exactly this assignment.
        """
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value.

        Raises
        ------
        ValueError
            If the value is not a finite real number. A ``NaN`` that reaches OpenMDAO
            propagates silently through a Newton solve and surfaces as a convergence failure
            with no explanation, so it is rejected here instead.
        """
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Parameter '{self._name}' was given a non-numeric value {value!r}.") from error
        if not math.isfinite(number):
            raise ValueError(f"Parameter '{self._name}' was given a non-finite value {number!r}.")
        self._value = number

    # -- derived -------------------------------------------------------------------------

    def replace(self, value: float) -> Parameter:
        """Return a copy of this parameter carrying a different value.

        Parameters
        ----------
        value : float
            Value of the copy.

        Returns
        -------
        Parameter
            A new parameter with the same name, units and source.
        """
        return Parameter(self._name, value, self._units, self._source)

    def __eq__(self, other: object) -> bool:
        """Return whether two parameters agree in name, value, units and source."""
        if not isinstance(other, Parameter):
            return NotImplemented
        return (self._name, self._value, self._units, self._source) == (
            other._name,
            other._value,
            other._units,
            other._source,
        )

    def __hash__(self) -> int:
        """Return a hash consistent with :meth:`__eq__`."""
        return hash((self._name, self._value, self._units, self._source))

    def __repr__(self) -> str:
        """Return a representation naming the variable, its value and its units."""
        return f"Parameter({self._name!r}, {self._value!r}, {self._units!r})"


class Response:
    """One quantity read back out of the black box.

    Parameters
    ----------
    name : str
        Short name cdadt reports it under, e.g. ``"block_fuel"``. Unique within a discipline.
    path : str
        Where the quantity lives inside the black box, e.g.
        ``"mission.descent.fuel_burn_integ.fuel_burn_final"``. This is the black box's name,
        not cdadt's, and it is written down here rather than assembled at read time so that
        the whole interface can be listed without running anything.
    units : str or None, optional
        Units to read the quantity in. ``None`` for dimensionless.
    description : str, optional
        One line explaining what the quantity is.
    optional : bool, optional
        Whether the quantity may be absent. Default ``False``. A response is optional when it
        exists only in a particular black-box model -- the per-component weight breakdown, for
        instance, exists because OpenConcept's jet-transport empty-weight buildup publishes it,
        and a different sizing model need not. A missing *required* response is an error; a
        missing optional one is reported as unavailable.

    Examples
    --------
    >>> Response("block_fuel", "mission.descent.fuel_burn_integ.fuel_burn_final", "kg")
    Response('block_fuel', 'mission.descent.fuel_burn_integ.fuel_burn_final', 'kg')
    """

    __slots__ = ("_description", "_name", "_optional", "_path", "_units")

    def __init__(
        self,
        name: str,
        path: str,
        units: str | None = None,
        description: str = "",
        optional: bool = False,
    ) -> None:
        if not name:
            raise ValueError("A response must have a name.")
        if not path:
            raise ValueError(f"Response '{name}' must name a path inside the black box.")
        self._name = str(name)
        self._path = str(path)
        self._units = units
        self._description = str(description)
        self._optional = bool(optional)

    @property
    def name(self) -> str:
        """Short name cdadt reports the quantity under."""
        return self._name

    @property
    def path(self) -> str:
        """Path of the quantity inside the black box."""
        return self._path

    @property
    def units(self) -> str | None:
        """Units the quantity is read in, or ``None`` if dimensionless."""
        return self._units

    @property
    def description(self) -> str:
        """One line explaining what the quantity is."""
        return self._description

    @property
    def optional(self) -> bool:
        """Whether the quantity may legitimately be absent from a given black-box model."""
        return self._optional

    def __eq__(self, other: object) -> bool:
        """Return whether two responses name the same quantity in the same way."""
        if not isinstance(other, Response):
            return NotImplemented
        return (self._name, self._path, self._units, self._optional) == (
            other._name,
            other._path,
            other._units,
            other._optional,
        )

    def __hash__(self) -> int:
        """Return a hash consistent with :meth:`__eq__`."""
        return hash((self._name, self._path, self._units, self._optional))

    def __repr__(self) -> str:
        """Return a representation naming the response, its path and its units."""
        return f"Response({self._name!r}, {self._path!r}, {self._units!r})"
