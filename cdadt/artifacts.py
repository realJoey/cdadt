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

__all__ = ["ArtifactError", "MissionTrajectory", "StudyArtifacts", "TakeoffTrajectory", "Trace"]


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


class TakeoffTrajectory:
    """The balanced field: what the aeroplane does before the mission starts.

    Neither reference example plots this -- ``B738.py`` and ``B738_sizing.py`` both list the same
    seven steady phases -- but the takeoff is where the field length comes from, and a balanced
    field is a statement about two paths that a table of numbers cannot show.

    **Two paths, not one.** From brake release the aeroplane accelerates to V\\ :sub:`1`
    (``v0v1``). At V\\ :sub:`1` it either continues, accelerating to rotation and lifting off
    (``v1vr`` then ``rotate``), or it rejects and brakes to a stop (``v1v0``). Balanced means the
    two end at the same distance. Drawn together against ground distance, that is the picture;
    concatenated into one line, as the mission phases are, it would be a trajectory that doubles
    back on itself, which is exactly why :class:`MissionTrajectory` leaves these out.

    **Why these four are named rather than discovered.** The mission phases are discovered
    because nothing distinguishes them but their structure. These four are different: which one
    is the continued takeoff and which is the rejected one is *semantics*, and no output the box
    publishes carries it. The names are OpenConcept's own, from the takeoff group that
    ``B738_sizing.py`` builds, and a box that does not have them raises rather than guessing.

    Parameters
    ----------
    box : OpenConceptSizingBox
        A converged box.
    mission_path : str, optional
        Name of the mission subsystem. Default ``"mission"``.

    Raises
    ------
    ArtifactError
        If the box does not publish the ground-roll phases, naming the ones it is missing.
    """

    #: Accelerate to V1, then continue: rotate and lift off.
    CONTINUE: ClassVar[tuple[str, ...]] = ("v0v1", "v1vr", "rotate")

    #: Accelerate to V1, then reject and brake to a stop.
    ABORT: ClassVar[tuple[str, ...]] = ("v0v1", "v1v0")

    ABSCISSA: ClassVar[Trace] = Trace("range", "ft", "Ground distance (ft)")

    TRACES: ClassVar[tuple[Trace, ...]] = (
        Trace("fltcond|Utrue", "kn", "True airspeed (knots)"),
        Trace("fltcond|h", "ft", "Altitude (ft)"),
        Trace("throttle", None, "Throttle setting"),
        Trace("weight", "lb", "Weight (lb)"),
    )

    #: Where the decision speed lives, relative to the mission.
    V1_PATH: ClassVar[str] = "takeoff|v1"

    def __init__(self, box: Any, mission_path: str = "mission") -> None:
        self._box = box
        self._mission_path = str(mission_path)
        missing = [
            phase
            for phase in dict.fromkeys((*self.CONTINUE, *self.ABORT))
            if not box.has(f"{self._mission_path}.{phase}.{self.ABSCISSA.name}")
        ]
        if missing:
            raise ArtifactError(
                f"The black box publishes no ground roll for {missing} under "
                f"'{self._mission_path}', so there is no takeoff to plot. These are "
                f"OpenConcept's own takeoff phase names; a mission model without a balanced "
                f"field has no takeoff figure."
            )

    @property
    def decision_speed(self) -> float:
        """V\\ :sub:`1`, in knots: where the two paths separate."""
        return float(np.asarray(self._box.get(f"{self._mission_path}.{self.V1_PATH}", units="kn")).reshape(-1)[0])

    def path(self, phases: Sequence[str], trace: Trace) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distance, values)`` along one of the two paths, concatenated in order."""
        distance, values = [], []
        for phase in phases:
            distance.append(self._read(phase, self.ABSCISSA))
            values.append(self._read(phase, trace))
        return np.concatenate(distance), np.concatenate(values)

    def _read(self, phase: str, trace: Trace) -> np.ndarray:
        """Return one trace from one phase, as a one-dimensional array."""
        path = f"{self._mission_path}.{phase}.{trace.name}"
        return np.atleast_1d(np.asarray(self._box.get(path, units=trace.units), dtype=float)).ravel()

    @property
    def available(self) -> tuple[Trace, ...]:
        """The traces every ground-roll phase publishes."""
        return tuple(
            trace
            for trace in self.TRACES
            if all(
                self._box.has(f"{self._mission_path}.{phase}.{trace.name}") for phase in (*self.CONTINUE, *self.ABORT)
            )
        )

    def __repr__(self) -> str:
        """Return a representation naming the two paths."""
        return f"TakeoffTrajectory(continue={list(self.CONTINUE)}, abort={list(self.ABORT)})"


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

    #: The panels ``B738_sizing.py``'s own ``plot_results`` draws, in its order and its units.
    #: The fifth carries two series on one axis, which is why a panel is a tuple of traces.
    PROFILE_PANELS: ClassVar[tuple[tuple[str, tuple[Trace, ...]], ...]] = (
        ("Altitude (ft)", (Trace("fltcond|h", "ft", "Altitude"),)),
        ("Mach number", (Trace("fltcond|M", None, "Mach"),)),
        ("Vertical speed (ft/min)", (Trace("fltcond|vs", "ft/min", "Vertical speed"),)),
        ("Weight (lb)", (Trace("weight", "lb", "Weight"),)),
        (
            "Longitudinal force (lb)",
            (Trace("drag", "lbf", "Drag"), Trace("thrust", "lbf", "Thrust")),
        ),
        ("Throttle (%)", (Trace("throttle", None, "Throttle"),)),
    )

    def write_mission_profile(
        self,
        box: Any,
        title: str,
        mission_path: str = "mission",
        name: str = "mission.pdf",
    ) -> Path:
        """Reproduce ``B738_sizing.py``'s ``plot_results``: the 2x3 mission profile.

        This is the figure belonging to the example cdadt actually drives, as distinct from
        :meth:`write_trajectory`, which reproduces the seven-panel figure ``B738.py`` draws in
        ``show_outputs``. They plot different quantities, so cdadt writes both rather than
        choosing which example to be faithful to.

        Throttle is drawn as a percentage and the force panel carries drag and thrust together,
        both as ``plot_results`` does it.

        Raises
        ------
        ArtifactError
            If matplotlib is not installed, or if the box publishes no trajectory.
        """
        trajectory = MissionTrajectory(box, mission_path=mission_path)
        plt = self._pyplot()

        figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), squeeze=False)
        flat = [axis for row in axes for axis in row]

        for axis, (label, traces) in zip(flat, self.PROFILE_PANELS, strict=True):
            for trace in traces:
                values = trajectory.read(trace)
                axis.plot(trajectory.read(trajectory.ABSCISSA), values * 100.0 if "%" in label else values, "-")
            axis.set_xlabel("Distance flown (nmi)")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            if len(traces) > 1:
                axis.legend([trace.label for trace in traces])

        figure.suptitle(title)
        figure.tight_layout()
        destination = self._directory / name
        figure.savefig(destination)
        plt.close(figure)
        return destination

    def write_takeoff(
        self,
        box: Any,
        title: str,
        mission_path: str = "mission",
        name: str = "takeoff.pdf",
    ) -> Path:
        """Plot the balanced field: the continued takeoff and the rejected one, against distance.

        Neither reference example draws this. It is here because the field length is what the
        takeoff phases are *for*, and "balanced" is a claim about two paths ending at the same
        distance -- which a number in a table states but does not show.

        Raises
        ------
        ArtifactError
            If matplotlib is not installed, or if the box has no ground roll.
        """
        takeoff = TakeoffTrajectory(box, mission_path=mission_path)
        plt = self._pyplot()

        traces = takeoff.available
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), squeeze=False)
        flat = [axis for row in axes for axis in row]

        for axis, trace in zip(flat, traces, strict=False):
            for phases, style, label in (
                (takeoff.CONTINUE, "-", "Continue"),
                (takeoff.ABORT, "--", "Reject"),
            ):
                distance, values = takeoff.path(phases, trace)
                axis.plot(distance, values, style, label=label)
            axis.set_xlabel(takeoff.ABSCISSA.label)
            axis.set_ylabel(trace.label)
            axis.grid(True, alpha=0.3)
            axis.legend()
        for axis in flat[len(traces) :]:
            axis.axis("off")

        figure.suptitle(f"{title} -- balanced field, V1 = {takeoff.decision_speed:.1f} kn")
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
        except ImportError as error:
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
