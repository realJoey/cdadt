"""What a run produced, and how to name a quantity without knowing where it lives.

Two classes. :class:`ResponseCatalog` maps the short names cdadt reports under --
``total_fuel``, ``MTOW``, ``takeoff_field_length`` -- onto the paths those quantities live at
inside the black box, so that a case file can write ``objective: {name: total_fuel}`` instead of
``mission.loiter.fuel_burn_integ.fuel_burn_final``. The catalog is built from the discipline
classes themselves, so a response added to a discipline is addressable in a case file
immediately and nothing has to be kept in step by hand.

:class:`SizingResults` is the answer: every response of every discipline, read on every run
rather than whichever few a caller thought to ask for. Reading the whole set is what makes a run
report complete, and what makes two runs comparable without re-running either.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from cdadt.disciplines import AIRCRAFT_DISCIPLINES, Discipline, Performance
from cdadt.parameters import Response

__all__ = ["ResponseCatalog", "SizingResults"]


class ResponseCatalog:
    """Every response the disciplines of a study report, by short name.

    Parameters
    ----------
    disciplines : sequence of type or Discipline, optional
        Discipline classes or instances to catalogue. Default: the airframe disciplines plus
        performance.

    Raises
    ------
    ValueError
        If two disciplines report different quantities under the same name. Names are how a
        case file addresses a response, so they have to be unique across the study.

    Examples
    --------
    >>> catalog = ResponseCatalog()
    >>> catalog.path("total_fuel")
    'mission.loiter.fuel_burn_integ.fuel_burn_final'
    >>> catalog.units("takeoff_field_length")
    'ft'
    """

    def __init__(self, disciplines: Sequence[Any] | None = None) -> None:
        sources = disciplines if disciplines is not None else (*AIRCRAFT_DISCIPLINES, Performance)
        self._responses: dict[str, Response] = {}
        self._owners: dict[str, str] = {}
        for source in sources:
            name = source.discipline_name
            for response in source.reported:
                existing = self._responses.get(response.name)
                if existing is not None and existing != response:
                    raise ValueError(
                        f"'{response.name}' is reported by both '{self._owners[response.name]}' and "
                        f"'{name}', with different definitions."
                    )
                self._responses[response.name] = response
                self._owners[response.name] = name

    def __contains__(self, name: str) -> bool:
        """Return whether a response of that name is catalogued."""
        return name in self._responses

    def __iter__(self) -> Iterator[str]:
        """Iterate the catalogued response names."""
        return iter(self._responses)

    def __len__(self) -> int:
        """Return the number of catalogued responses."""
        return len(self._responses)

    def response(self, name: str) -> Response:
        """Return one catalogued response.

        Raises
        ------
        KeyError
            If nothing reports it, listing what is available.
        """
        try:
            return self._responses[name]
        except KeyError:
            raise KeyError(
                f"'{name}' is not a response any discipline reports. Available: {sorted(self._responses)}."
            ) from None

    def path(self, name: str) -> str:
        """Return the black-box path of a response, or ``name`` itself if it is already a path.

        A case file may name either. Anything the catalog does not know is passed through
        unchanged, so a raw path always works and the black box gives the error if it is wrong.
        """
        return self._responses[name].path if name in self._responses else name

    def units(self, name: str) -> str | None:
        """Return the units of a response, or ``None`` for a raw path or a dimensionless one."""
        return self._responses[name].units if name in self._responses else None

    def owner(self, name: str) -> str:
        """Return the name of the discipline that reports ``name``."""
        return self._owners[name]

    def __repr__(self) -> str:
        """Return a representation naming the catalogue size."""
        return f"ResponseCatalog({len(self._responses)} responses)"


class SizingResults:
    """Everything one converged run produced.

    Parameters
    ----------
    values : mapping
        ``discipline name -> {response name -> value}``.
    missing : mapping, optional
        ``discipline name -> optional response names the black box did not publish``. Recorded
        rather than dropped: a report that silently omits a quantity reads as a report that
        found nothing to say about it.
    model : str, optional
        The ``module:Class`` string of the black box that produced these numbers.
    num_nodes : int, optional
        Analysis points per phase the run used.

    Examples
    --------
    >>> results["total_fuel"]
    18597.3...
    >>> results.by_discipline["weights"]["MTOW"]
    78345.6...
    """

    def __init__(
        self,
        values: Mapping[str, Mapping[str, Any]],
        missing: Mapping[str, Sequence[str]] | None = None,
        model: str = "",
        num_nodes: int = 0,
    ) -> None:
        self._values = {name: dict(entries) for name, entries in values.items()}
        self._missing = {name: tuple(entries) for name, entries in (missing or {}).items()}
        self._model = str(model)
        self._num_nodes = int(num_nodes)

        self._flat: dict[str, Any] = {}
        for entries in self._values.values():
            self._flat.update(entries)

    # -- state ---------------------------------------------------------------------------

    @property
    def by_discipline(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the results grouped by discipline."""
        return {name: dict(entries) for name, entries in self._values.items()}

    @property
    def missing(self) -> Mapping[str, tuple[str, ...]]:
        """Return the optional responses the black box did not publish, by discipline."""
        return dict(self._missing)

    @property
    def model(self) -> str:
        """The black box that produced these numbers."""
        return self._model

    @property
    def num_nodes(self) -> int:
        """Analysis points per phase the run used."""
        return self._num_nodes

    def __getitem__(self, name: str) -> Any:
        """Return one response by name, from whichever discipline reported it."""
        try:
            return self._flat[name]
        except KeyError:
            raise KeyError(f"'{name}' was not reported by this run. Reported: {sorted(self._flat)}.") from None

    def __contains__(self, name: str) -> bool:
        """Return whether a response of that name was reported."""
        return name in self._flat

    def __iter__(self) -> Iterator[str]:
        """Iterate every reported response name."""
        return iter(self._flat)

    def __len__(self) -> int:
        """Return the number of reported responses."""
        return len(self._flat)

    def scalars(self) -> dict[str, float]:
        """Return only the single-valued results, as floats.

        Vector results -- throttle histories and the like -- are omitted, because the things a
        report tabulates and an optimizer constrains are scalars.
        """
        return {name: float(value) for name, value in self._flat.items() if np.asarray(value).size == 1}

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, serializable dictionary of the whole run."""
        return {
            "model": self._model,
            "num_nodes": self._num_nodes,
            "results": {
                discipline: {
                    name: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for name, value in entries.items()
                }
                for discipline, entries in self._values.items()
            },
            "unavailable": {discipline: list(names) for discipline, names in self._missing.items()},
        }

    # -- reporting -----------------------------------------------------------------------

    def report(self, catalog: ResponseCatalog | None = None) -> str:
        """Return the run as a table, grouped by discipline.

        Parameters
        ----------
        catalog : ResponseCatalog, optional
            Used for units. Defaults to a catalogue of the standard disciplines.
        """
        catalog = catalog if catalog is not None else ResponseCatalog()
        lines = [
            f"Black box : {self._model}",
            f"Grid      : {self._num_nodes} nodes per phase",
            "",
        ]
        for discipline, entries in self._values.items():
            if not entries:
                continue
            lines.append(discipline)
            lines.append("-" * len(discipline))
            for name, value in entries.items():
                units = catalog.units(name) or "-"
                array = np.asarray(value)
                if array.size == 1:
                    lines.append(f"  {name:<28s} {float(array.reshape(-1)[0]):16.4f}  {units}")
                else:
                    lines.append(
                        f"  {name:<28s} {array.min():7.4f} to {array.max():<7.4f} ({array.size} nodes)  {units}"
                    )
            lines.append("")

        if self._missing:
            lines.append("Not published by this black box")
            lines.append("-------------------------------")
            for discipline, names in self._missing.items():
                lines.append(f"  {discipline}: {', '.join(names)}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def collect(
        cls,
        box: Any,
        disciplines: Sequence[Discipline],
    ) -> SizingResults:
        """Read every discipline's responses out of a converged box.

        Parameters
        ----------
        box : OpenConceptSizingBox
            A built and converged box.
        disciplines : sequence of Discipline
            The discipline instances to read.
        """
        values = {d.discipline_name: d.collect(box) for d in disciplines}
        missing = {d.discipline_name: d.missing(box) for d in disciplines if d.missing(box)}
        return cls(values, missing, model=box.model_spec, num_nodes=box.num_nodes)

    def __repr__(self) -> str:
        """Return a representation naming how much was reported."""
        return f"SizingResults({len(self._values)} disciplines, {len(self._flat)} responses)"
