"""The files a study leaves behind: the model diagram, the flown trajectory, the report.

A run that prints its numbers and exits leaves nothing to attach to a design review. This module
writes the three artefacts a study is normally asked for, into one directory the caller names:

``n2.html``
    OpenMDAO's N2 diagram of the built model. Every component, every connection, every solver --
    generated from the live problem, so it is the model that was actually run.

``trajectory.pdf``
    What was flown, plotted against range. The quantities and their units are the ones
    OpenConcept's own ``B738.py`` plots in ``show_outputs``: altitude, equivalent airspeed, fuel
    used, throttle, vertical speed, Mach number and lift coefficient.

``report.txt``, ``results.json``
    The same text the command line prints, and the same numbers as structured data.

Nothing here computes anything. The trajectory is read out of the converged box phase by phase,
in the units the box is asked for them in, exactly as every other cdadt response is read -- and
the phases are *discovered* from what the box publishes rather than listed, so a different
mission model plots correctly without being described here.

Matplotlib is an optional dependency (``pip install -e ".[plot]"``). Only :meth:`
StudyArtifacts.write_trajectory` needs it, and it says so by name if it is missing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

__all__ = ["ArtifactError", "MissionTrajectory", "StudyArtifacts", "Trace"]


class ArtifactError(Exception):
    """Raised when an artefact cannot be produced from the box as it stands."""


class Trace:
    """One quantity plotted along the mission, read from every phase and concatenated.

    Parameters
    ----------
    name : str
        Name of the variable inside a phase, e.g. ``"fltcond|h"``.
    units : str or None
        Units to read it in. ``None`` for a dimensionless quantity such as throttle.
    label : str
        Axis label, including the unit as a reader expects to see it.

    Examples
    --------
    >>> Trace("fltcond|h", "ft", "Altitude (ft)")
    Trace('fltcond|h', 'ft')
    """

    __slots__ = ("_label", "_name", "_units")

    def __init__(self, name: str, units: str | None, label: str) -> None:
        if not name:
            raise ValueError("A trace must name a variable.")
        self._name = str(name)
        self._units = units
        self._label = str(label)

    @property
    def name(self) -> str:
        """Name of the variable inside a phase."""
        return self._name

    @property
    def units(self) -> str | None:
        """Units the variable is read in."""
        return self._units

    @property
    def label(self) -> str:
        """Axis label."""
        return self._label

    def path(self, mission_path: str, phase: str) -> str:
        """Return the full black-box path of this trace within one phase."""
        return f"{mission_path}.{phase}.{self._name}"

    def __repr__(self) -> str:
        """Return a representation naming the variable and its units."""
        return f"Trace({self._name!r}, {self._units!r})"


class MissionTrajectory:
    """What was flown, read out of a converged black box.

    The phases are discovered rather than listed, and ordered by where they happen rather than by
    what they are called.

    **Which subsystems are phases.** A steady flight phase in an OpenConcept mission integrates
    its own state and therefore publishes ``ode_integ_phase.range_final``; the ground-roll phases
    -- ``v0v1``, ``v1vr``, ``rotate`` and the rejected-takeoff ``v1v0`` -- do not. That is the
    discriminator, and being structural rather than a list of names it lets a mission model with
    a second diversion, or without a loiter, plot correctly with nothing here changed. It also
    keeps the rejected takeoff out of the picture: ``v1v0`` retraces ground already covered, and
    drawn on a range axis it is a trajectory that doubles back.

    **What order they are flown in.** Alphabetical order is not flight order -- it puts loiter
    before the reserve climb -- and the box publishes its outputs sorted by name, so model order
    is not available to be read either. Phases are therefore ordered by the abscissa itself:
    where each one *starts*. That is a fact about the converged mission rather than a convention.

    Parameters
    ----------
    box : OpenConceptSizingBox
        A **converged** box. Reading an unconverged one produces a plot of a state the solver was
        passing through, which is indistinguishable from a mission -- and leaves the phase order
        resting on numbers that do not mean anything yet.
    mission_path : str, optional
        Name of the mission subsystem inside the box. Default ``"mission"``.

    Raises
    ------
    ArtifactError
        If the box publishes no steady flight phase, which means either that it was never built
        or that ``mission_path`` names the wrong subsystem.

    Attributes
    ----------
    ABSCISSA : Trace
        What everything is plotted against: ground distance covered.
    PHASE_MARKER : str
        The output whose presence makes a mission subsystem a steady flight phase.
    TRACES : tuple of Trace
        The plotted quantities: the ones OpenConcept's own ``B738.py`` plots, in its order and
        its units, with one difference in *where the number comes from*. It plots ``fuel_used``,
        which is a promoted *input* of the phase and therefore not unambiguously readable. What
        the box publishes as an output is the phase's fuel integrator, and each phase's carries
        on from the end of the one before rather than restarting -- measured, not assumed: on the
        shipped case the last node reads 41000.07 lbm, which is ``total_fuel`` to all printed
        digits. The panel is therefore cumulative mission fuel, as ``B738.py`` plots it.
    """

    ABSCISSA: ClassVar[Trace] = Trace("range", "NM", "Range (nmi)")

    PHASE_MARKER: ClassVar[str] = "ode_integ_phase.range_final"

    TRACES: ClassVar[tuple[Trace, ...]] = (
        Trace("fltcond|h", "ft", "Altitude (ft)"),
        Trace("fltcond|Ueas", "kn", "Veas airspeed (knots)"),
        Trace("fuel_burn_integ.fuel_burn", "lbm", "Fuel burned (lb)"),
        Trace("throttle", None, "Throttle setting"),
        Trace("fltcond|vs", "ft/min", "Vertical speed (ft/min)"),
        Trace("fltcond|M", None, "Mach number"),
        Trace("fltcond|CL", None, "CL"),
    )

    def __init__(self, box: Any, mission_path: str = "mission") -> None:
        self._box = box
        self._mission_path = str(mission_path)
        self._phases = self._discover()
        if not self._phases:
            raise ArtifactError(
                f"The black box publishes no steady flight phase under '{self._mission_path}', "
                f"so there is no trajectory to plot. A phase is recognised by publishing "
                f"'{self.PHASE_MARKER}'. Check that the box is built and that 'mission_path' "
                f"names the right subsystem."
            )

    def _discover(self) -> tuple[str, ...]:
        """Return the steady flight phases, ordered by where each one starts.

        Discovery reads :meth:`~cdadt.blackbox.OpenConceptSizingBox.readable` and must keep
        doing so. ``has`` is not the same question and would give the wrong answer here: it falls
        back to ``get_val``, and on the shipped model ``v0v1``, ``v1vr`` and ``v1v0`` all answer
        it for the marker even though none of them publishes the marker as an output. Asking
        ``has`` instead would put the rejected takeoff back into the trajectory, where it doubles
        back on the range axis. Only ``rotate`` is excluded by both.
        """
        prefix = f"{self._mission_path}."
        suffix = f".{self.PHASE_MARKER}"
        found = [
            path[len(prefix) : -len(suffix)]
            for path in self._box.readable()
            if path.startswith(prefix) and path.endswith(suffix)
        ]
        # Name as the tie-break, so the order is still deterministic on an unconverged box where
        # every phase starts at the same meaningless zero.
        return tuple(sorted(found, key=lambda phase: (self._start_of(phase), phase)))

    def _start_of(self, phase: str) -> float:
        """Return the abscissa at the first node of ``phase``."""
        path = self.ABSCISSA.path(self._mission_path, phase)
        return float(np.atleast_1d(np.asarray(self._box.get(path, units=self.ABSCISSA.units))).ravel()[0])

    @property
    def phases(self) -> tuple[str, ...]:
        """The phases flown, in flight order."""
        return self._phases

    @property
    def available(self) -> tuple[Trace, ...]:
        """The traces every phase publishes, and which can therefore be plotted whole.

        A trace one phase is missing is dropped rather than plotted with a gap, because a
        trajectory drawn across a hole reads as a trajectory, not as missing data.
        """
        return tuple(trace for trace in self.TRACES if self._is_whole(trace))

    def _is_whole(self, trace: Trace) -> bool:
        """Return whether every phase publishes ``trace``."""
        return all(self._box.has(trace.path(self._mission_path, phase)) for phase in self._phases)

    def read(self, trace: Trace) -> np.ndarray:
        """Return one trace, concatenated across the phases in flight order.

        Raises
        ------
        ArtifactError
            If a phase does not publish it, naming the phase.
        """
        segments = []
        for phase in self._phases:
            path = trace.path(self._mission_path, phase)
            if not self._box.has(path):
                raise ArtifactError(f"Phase '{phase}' does not publish '{trace.name}'.")
            segments.append(np.atleast_1d(np.asarray(self._box.get(path, units=trace.units), dtype=float)))
        return np.concatenate(segments)

    def __len__(self) -> int:
        """Return the number of plottable traces."""
        return len(self.available)

    def __iter__(self) -> Iterator[tuple[Trace, np.ndarray, np.ndarray]]:
        """Iterate ``(trace, abscissa, values)`` for every trace that can be plotted whole."""
        abscissa = self.read(self.ABSCISSA)
        for trace in self.available:
            yield trace, abscissa, self.read(trace)

    def __repr__(self) -> str:
        """Return a representation naming the phase and trace counts."""
        return f"MissionTrajectory({len(self._phases)} phases, {len(self.available)} traces)"


class StudyArtifacts:
    """The directory a study writes its files into, and the files it writes.

    Parameters
    ----------
    directory : str or Path
        Where to write. Created if it does not exist, including parents.

    Examples
    --------
    >>> artifacts = StudyArtifacts("b738_out")
    >>> artifacts.write_text(report, "report.txt")
    PosixPath('b738_out/report.txt')
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Where the files are written."""
        return self._directory

    def write_text(self, text: str, name: str) -> Path:
        """Write one text file and return its path."""
        destination = self._directory / name
        destination.write_text(text, encoding="utf-8")
        return destination

    def write_json(self, payload: Any, name: str) -> Path:
        """Write one JSON file and return its path."""
        destination = self._directory / name
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def write_n2(self, box: Any, name: str = "n2.html") -> Path:
        """Write OpenMDAO's N2 diagram of the built model.

        Raises
        ------
        ArtifactError
            If the box has not been built, since there is no model to diagram.
        """
        if not box.is_built:
            raise ArtifactError("The black box has not been built, so there is no model to diagram.")
        import openmdao.api as om

        destination = self._directory / name
        om.n2(box.problem, outfile=str(destination), show_browser=False)
        return destination

    def write_trajectory(
        self,
        box: Any,
        title: str,
        mission_path: str = "mission",
        name: str = "trajectory.pdf",
        columns: int = 2,
    ) -> Path:
        """Plot the flown trajectory against range, one panel per quantity.

        Parameters
        ----------
        box : OpenConceptSizingBox
            A converged box.
        title : str
            Figure title, e.g. the case file's name.
        mission_path : str, optional
            Name of the mission subsystem. Default ``"mission"``.
        name : str, optional
            File to write. The extension decides the format, as matplotlib reads it.
        columns : int, optional
            Panels across. Default 2.

        Raises
        ------
        ArtifactError
            If matplotlib is not installed, or if the box publishes no trajectory.
        """
        trajectory = MissionTrajectory(box, mission_path=mission_path)
        plt = self._pyplot()

        traces = list(trajectory)
        rows = -(-len(traces) // columns)
        figure, axes = plt.subplots(rows, columns, figsize=(11.0, 2.6 * rows), squeeze=False)
        flat = [axis for row in axes for axis in row]

        # Not strict: there are deliberately more axes than traces whenever the panel count does
        # not divide evenly, and the spare ones are switched off below.
        for axis, (trace, abscissa, values) in zip(flat, traces, strict=False):
            axis.plot(abscissa, values, "-")
            axis.set_xlabel(trajectory.ABSCISSA.label)
            axis.set_ylabel(trace.label)
            axis.grid(True, alpha=0.3)
        for axis in flat[len(traces) :]:
            axis.axis("off")

        figure.suptitle(title)
        figure.tight_layout()
        destination = self._directory / name
        figure.savefig(destination)
        plt.close(figure)
        return destination

    @staticmethod
    def _pyplot() -> Any:
        """Return ``matplotlib.pyplot``, on a non-interactive backend.

        The backend is forced because a study is normally run headless, and the interactive
        default blocks on a window nobody is there to close.

        Raises
        ------
        ArtifactError
            If matplotlib is not installed, naming the extra that provides it.
        """
        try:
            import matplotlib
        except ImportError as error:  # pragma: no cover - depends on the installed stack
            raise ArtifactError(
                "Plotting the trajectory needs matplotlib, which is an optional dependency. "
                'Install it with: pip install -e ".[plot]"'
            ) from error
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt

    def written(self) -> Sequence[Path]:
        """Return every file in the directory, sorted, so a caller can report what it produced."""
        return sorted(path for path in self._directory.iterdir() if path.is_file())

    def __repr__(self) -> str:
        """Return a representation naming the directory."""
        return f"StudyArtifacts({str(self._directory)!r})"
