"""The certification basis: regulations as first-class modeling objects.

This is what makes cdadt certification-*driven* rather than certification-*checked*. A
:class:`Requirement` is not a constraint expression buried in a run script. It is an object
that knows the regulation it enforces, the model variables it reads, the components it
contributes, the constraint it registers with the optimizer, and how to report its own
margin afterwards.

The consequence is the traceability matrix: after any run, every active regulation can be
listed alongside the constraint that enforced it, the value achieved, the limit, the margin,
and the source of the limit. That table is the artifact a certification-driven design method
has to produce. A design that merely converged does not tell you which requirements shaped
it.

Every limit is configured, never defaulted. A required climb gradient or a field length is a
number that comes from a regulation *and an operating case* -- a runway, an altitude, a
temperature. Building one into the code applies it to aircraft nobody chose it for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import openmdao.api as om

from cdadt.core.configuration import AircraftConfiguration
from cdadt.mission.blackbox import MissionBlackBox

__all__ = ["CertificationBasis", "Requirement", "RequirementResult", "Sense"]


class Sense:
    """Which side of the limit satisfies a requirement.

    Attributes
    ----------
    UPPER : str
        The quantity must not exceed the limit, e.g. a balanced field length.
    LOWER : str
        The quantity must not fall below the limit, e.g. a climb gradient.
    """

    UPPER = "upper"
    LOWER = "lower"


@dataclass(frozen=True)
class RequirementResult:
    """The outcome of one requirement, as it appears in the traceability matrix.

    Parameters
    ----------
    requirement : Requirement
        The requirement this result belongs to.
    value : float
        The value achieved by the design.
    limit : float
        The limit the requirement enforces.
    units : str or None
        Units of both value and limit.
    satisfied : bool
        Whether the design meets the requirement.
    margin : float
        Signed margin in the requirement's units: positive when satisfied, and zero when
        the requirement is exactly binding. For an upper limit this is ``limit - value``;
        for a lower limit, ``value - limit``.
    """

    requirement: Requirement
    value: float
    limit: float
    units: str | None
    satisfied: bool
    margin: float

    @property
    def relative_margin(self) -> float:
        """Return the margin as a fraction of the limit.

        Returns
        -------
        float
            ``margin / abs(limit)``, or ``float("inf")`` if the limit is zero. Useful for
            ranking which requirements are actually driving a design: an absolute margin
            in feet and one in radians cannot be compared, but their relative margins can.
        """
        if self.limit == 0.0:
            return float("inf")
        return self.margin / abs(self.limit)


class Requirement(ABC):
    """One certification requirement, as an object.

    Parameters
    ----------
    config : AircraftConfiguration
        Configuration holding the limit and any method constants this requirement needs.

    Attributes
    ----------
    config : AircraftConfiguration
        The configuration passed at construction.

    Notes
    -----
    A requirement that only constrains a variable the mission already produces needs to
    implement :attr:`regulation`, :attr:`title`, :attr:`name`, :attr:`sense`,
    :meth:`limit`, and :meth:`constrained_path`. One that needs new physics -- landing
    performance, for instance, which OpenConcept does not model -- also overrides
    :meth:`build` to contribute components.
    """

    def __init__(self, config: AircraftConfiguration) -> None:
        self._config = config
        self.validate_configuration()

    @property
    def config(self) -> AircraftConfiguration:
        """Return the configuration this requirement reads."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short identifier, used as a subsystem name and as a report key."""

    @property
    @abstractmethod
    def regulation(self) -> str:
        """Return the regulation citation, e.g. ``"14 CFR 25.113"``."""

    @property
    @abstractmethod
    def title(self) -> str:
        """Return a one-line statement of what the regulation requires."""

    @property
    @abstractmethod
    def sense(self) -> str:
        """Return :attr:`Sense.UPPER` or :attr:`Sense.LOWER`."""

    @property
    @abstractmethod
    def units(self) -> str | None:
        """Return the units the constrained quantity and its limit are expressed in."""

    @abstractmethod
    def limit(self) -> float:
        """Return the configured limit, in :attr:`units`."""

    @abstractmethod
    def constrained_path(self, blackbox: MissionBlackBox) -> str:
        """Return the problem path of the quantity this requirement constrains.

        Parameters
        ----------
        blackbox : MissionBlackBox
            The mission, which knows where its own results live. Requirements ask it
            rather than hardcoding OpenConcept paths.

        Returns
        -------
        str
            A path usable with :meth:`openmdao.api.Problem.get_val`.
        """

    @property
    def limit_source(self) -> str | None:
        """Return the recorded provenance of the limit, for the traceability matrix."""
        return None

    def validate_configuration(self) -> None:  # noqa: B027 -- optional hook, deliberately not abstract
        """Check that the configuration holds this requirement's limit and constants.

        Subclasses call
        :meth:`~cdadt.core.configuration.AircraftConfiguration.require_all` here so that a
        configuration missing a limit fails at construction, rather than at solve time with
        a limit nobody chose.
        """

    def build(self, model: om.Group, blackbox: MissionBlackBox) -> None:  # noqa: B027 -- optional hook
        """Add any components this requirement needs to evaluate itself.

        The default adds nothing: most requirements constrain a quantity the mission
        already produces. Requirements covering conditions the mission does not model --
        landing, for one -- override this.

        Parameters
        ----------
        model : openmdao.api.Group
            The top-level model.
        blackbox : MissionBlackBox
            The mission, for building paths to quantities to read.
        """

    def register(self, model: om.Group, blackbox: MissionBlackBox) -> None:
        """Register this requirement's constraint with the optimizer.

        Parameters
        ----------
        model : openmdao.api.Group
            The model to register the constraint on.
        blackbox : MissionBlackBox
            The mission, for resolving the constrained path.
        """
        bound = {self.sense: self.limit()}
        model.add_constraint(self.constrained_path(blackbox), units=self.units, **bound)

    def evaluate(self, problem: om.Problem, blackbox: MissionBlackBox) -> RequirementResult:
        """Read the achieved value from a run problem and report the margin.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A converged problem.
        blackbox : MissionBlackBox
            The mission, for resolving the constrained path.

        Returns
        -------
        RequirementResult
            The value, the limit, and the signed margin.

        Notes
        -----
        For a vector quantity -- throttle across a phase, say -- the *worst* element is
        reported, since a requirement is satisfied only if it holds everywhere.
        """
        values = problem.get_val(self.constrained_path(blackbox), units=self.units)
        limit = self.limit()

        if self.sense == Sense.UPPER:
            value = float(max(values.reshape(-1)))
            margin = limit - value
        else:
            value = float(min(values.reshape(-1)))
            margin = value - limit

        return RequirementResult(
            requirement=self,
            value=value,
            limit=limit,
            units=self.units,
            satisfied=margin >= 0.0,
            margin=margin,
        )

    def __repr__(self) -> str:
        """Return a representation naming the regulation."""
        return f"{type(self).__name__}({self.regulation})"


class CertificationBasis:
    """The set of requirements a design is being certified against.

    Parameters
    ----------
    requirements : sequence of Requirement
        The active requirements.

    Raises
    ------
    ValueError
        If two requirements share a name, which would make the traceability matrix
        ambiguous about which one a row refers to.
    """

    def __init__(self, requirements: Sequence[Requirement]) -> None:
        self._requirements = tuple(requirements)
        names = [requirement.name for requirement in self._requirements]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Requirement names must be unique; duplicated: {duplicates}")

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        """Return the active requirements."""
        return self._requirements

    def __len__(self) -> int:
        """Return the number of active requirements."""
        return len(self._requirements)

    def __iter__(self):
        """Iterate the active requirements."""
        return iter(self._requirements)

    def build(self, model: om.Group, blackbox: MissionBlackBox) -> None:
        """Add every requirement's components to the model."""
        for requirement in self._requirements:
            requirement.build(model, blackbox)

    def register(self, model: om.Group, blackbox: MissionBlackBox) -> None:
        """Register every requirement's constraint with the optimizer."""
        for requirement in self._requirements:
            requirement.register(model, blackbox)

    def evaluate(self, problem: om.Problem, blackbox: MissionBlackBox) -> list[RequirementResult]:
        """Evaluate every requirement against a converged problem.

        Returns
        -------
        list of RequirementResult
            One result per requirement, in declaration order.
        """
        return [requirement.evaluate(problem, blackbox) for requirement in self._requirements]

    def traceability_matrix(self, problem: om.Problem, blackbox: MissionBlackBox) -> str:
        """Return the traceability matrix as a formatted table.

        Every active regulation appears with the constraint that enforced it, the value
        achieved, the limit, the margin, and where the limit came from. This is the
        artifact of a certification-driven method: it says which requirements shaped the
        design and by how much, which a converged result alone does not.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A converged problem.
        blackbox : MissionBlackBox
            The mission the requirements read from.

        Returns
        -------
        str
            A plain-text table, sorted with the most binding requirement first.
        """
        results = sorted(self.evaluate(problem, blackbox), key=lambda r: r.relative_margin)

        header = f"{'regulation':>16s}  {'requirement':<28s} {'value':>13s} {'limit':>13s} {'margin':>13s}  status"
        lines = [header, "-" * len(header)]
        for result in results:
            units = f" {result.units}" if result.units else ""
            lines.append(
                f"{result.requirement.regulation:>16s}  {result.requirement.title[:28]:<28s} "
                f"{result.value:13.4f} {result.limit:13.4f} {result.margin:13.4f}  "
                f"{'MET' if result.satisfied else 'NOT MET'}{units}"
            )

        lines.append("")
        lines.append("Limit sources:")
        for requirement in self._requirements:
            source = requirement.limit_source or "not recorded"
            lines.append(f"  {requirement.regulation}: {source}")

        unmet = [r for r in results if not r.satisfied]
        lines.append("")
        lines.append(f"{len(results) - len(unmet)} of {len(results)} requirements met.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation listing the regulations."""
        return f"CertificationBasis({[r.regulation for r in self._requirements]})"
