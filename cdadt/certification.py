"""Constraints, their provenance, and the traceability matrix.

A constraint here is what ``prob.model.add_constraint(...)`` is in OpenConcept's own optimizing
examples -- a bound on something the black box publishes -- plus two optional fields those
examples have nowhere to put: the regulation the number comes from, and the source of that
particular number.

.. code-block:: yaml

   constraints:
     - name: takeoff_field_length          # traceable to a regulation
       upper: 8000.0
       units: ft
       regulation: 14 CFR 25.113
       source: Design field length, 8000 ft dry runway at sea level, ISA

     - name: climb_throttle                # a plain bound, no provenance claimed
       lower: 0.01
       upper: 1.05

Both are legal. A constraint that names a regulation *and* a source appears in the traceability
matrix as a defensible requirement; one that does not is reported as a design constraint, and
the matrix says which is which rather than implying every row is certification evidence.

Everything else about a constraint -- the bound form, the units, the scaling, which elements of
a vector are constrained, whether it is linear -- is passed through to OpenMDAO unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import numpy as np
import openmdao.api as om

from cdadt.config import ConstraintSpec
from cdadt.results import ResponseCatalog

__all__ = ["CertificationBasis", "Constraint", "ConstraintError", "ConstraintResult"]


class ConstraintError(Exception):
    """Raised when a constraint cannot be registered or evaluated as stated."""


class Constraint:
    """One constraint, ready to register on a model and evaluate against a design.

    Parameters
    ----------
    spec : ConstraintSpec
        The case file's entry.
    catalog : ResponseCatalog, optional
        Used to turn a response name into a black-box path and supply its units. A name the
        catalogue does not know is passed through as a raw path.
    """

    def __init__(self, spec: ConstraintSpec, catalog: ResponseCatalog | None = None) -> None:
        self._spec = spec
        self._catalog = catalog if catalog is not None else ResponseCatalog()

    # -- state ---------------------------------------------------------------------------

    @property
    def spec(self) -> ConstraintSpec:
        """The case file's entry."""
        return self._spec

    @property
    def name(self) -> str:
        """The name the case file used, which is also the constraint's alias."""
        return self._spec.name

    @property
    def title(self) -> str:
        """One line naming the constraint in a report."""
        return self._spec.title

    @property
    def regulation(self) -> str:
        """The regulation this constraint comes from, or an empty string."""
        return self._spec.regulation

    @property
    def source(self) -> str:
        """Where the number came from, or an empty string."""
        return self._spec.source

    @property
    def is_traceable(self) -> bool:
        """Whether this constraint names both a regulation and a source."""
        return self._spec.is_traceable

    @property
    def path(self) -> str:
        """Where the constrained quantity lives inside the black box."""
        return self._catalog.path(self._spec.name)

    @property
    def units(self) -> str | None:
        """Units the constraint is evaluated in."""
        if self._spec.units is not None:
            return self._spec.units
        return self._catalog.units(self._spec.name)

    # -- use -----------------------------------------------------------------------------

    def register(self, model: om.Group) -> None:
        """Declare this constraint on the black box's group, before setup.

        Scaled by the magnitude of its own bound unless the case file says otherwise. Without
        that, a climb gradient in hundredths of a radian is numerically invisible next to a
        field length in thousands of feet.
        """
        arguments: dict[str, Any] = {
            "units": self.units,
            "alias": self.name,
            "linear": self._spec.linear,
            **self._spec.bounds.as_kwargs(),
            **self._spec.scaling.as_kwargs(default_ref=self._spec.bounds.magnitude),
        }
        if self._spec.indices is not None:
            arguments["indices"] = self._spec.indices
        model.add_constraint(self.path, **arguments)

    def evaluate(self, box: Any) -> ConstraintResult:
        """Read the constrained quantity and return the result.

        A vector response is one constraint, not one per node, and the node reported is the one
        that governs: the least margin, whichever side of the bound it is near.
        """
        values = np.atleast_1d(np.asarray(box.get(self.path, units=self.units)))
        if self._spec.indices is not None:
            values = values[self._spec.indices]

        bounds = self._spec.bounds
        if bounds.is_equality:
            governing = values[np.argmax(np.abs(values - np.asarray(bounds.equals)))]
        elif bounds.is_two_sided:
            margins = np.minimum(np.asarray(bounds.upper) - values, values - np.asarray(bounds.lower))
            governing = values[np.argmin(margins)]
        elif bounds.upper is not None:
            governing = values.max()
        else:
            governing = values.min()
        return ConstraintResult(self, float(governing))

    def __repr__(self) -> str:
        """Return a representation naming the constraint and its bound."""
        return f"Constraint({self.name!r}, {self._spec.bounds.describe()})"


class ConstraintResult:
    """One constraint, evaluated against one converged design.

    Attributes
    ----------
    ACTIVE_TOLERANCE : float
        Relative distance from the bound within which a constraint is called active rather than
        merely met. An active constraint is one that shaped the design.
    """

    ACTIVE_TOLERANCE: ClassVar[float] = 1e-4

    __slots__ = ("_constraint", "_value")

    def __init__(self, constraint: Constraint, value: float) -> None:
        self._constraint = constraint
        self._value = float(value)

    @property
    def constraint(self) -> Constraint:
        """The constraint that was checked."""
        return self._constraint

    @property
    def value(self) -> float:
        """The governing value, in the constraint's units."""
        return self._value

    @property
    def margin(self) -> float:
        """How far the design is on the satisfying side of its bound.

        Positive is compliant, whichever form the bound takes, so the sign means one thing
        across a whole matrix: an upper limit gives ``upper - value``; a lower limit gives
        ``value - lower``; a band gives whichever is smaller; an equality gives
        ``-|value - equals|``.
        """
        bounds = self._constraint.spec.bounds
        if bounds.is_equality:
            return -abs(self._value - float(np.atleast_1d(bounds.equals).ravel()[0]))
        margins = []
        if bounds.upper is not None:
            margins.append(float(np.min(np.atleast_1d(bounds.upper))) - self._value)
        if bounds.lower is not None:
            margins.append(self._value - float(np.max(np.atleast_1d(bounds.lower))))
        return min(margins)

    @property
    def relative_margin(self) -> float:
        """The margin as a fraction of the bound's magnitude."""
        magnitude = self._constraint.spec.bounds.magnitude
        return self.margin / magnitude if magnitude else float("inf")

    @property
    def satisfied(self) -> bool:
        """Whether the constraint is met."""
        return self.margin >= -self.ACTIVE_TOLERANCE * max(self._constraint.spec.bounds.magnitude, 1.0)

    @property
    def active(self) -> bool:
        """Whether the design sits on the bound, and was therefore shaped by it."""
        return self.satisfied and abs(self.relative_margin) <= self.ACTIVE_TOLERANCE

    @property
    def status(self) -> str:
        """``"MET"``, ``"ACTIVE"`` or ``"VIOLATED"``."""
        if not self.satisfied:
            return "VIOLATED"
        return "ACTIVE" if self.active else "MET"

    def __repr__(self) -> str:
        """Return a representation naming the constraint and its status."""
        return f"ConstraintResult({self._constraint.name!r}, {self._value:.4f}, {self.status})"


class CertificationBasis:
    """The set of constraints a design is held to, and the matrix that reports them.

    Parameters
    ----------
    constraints : sequence of Constraint
        What must hold. May be empty: an unconstrained study is a valid thing to run, and the
        report says so rather than printing an empty table.

    Raises
    ------
    ConstraintError
        If two constraints share a name, which OpenMDAO would resolve by keeping one of them.
    """

    def __init__(self, constraints: Sequence[Constraint] = ()) -> None:
        self._constraints = tuple(constraints)
        names = [constraint.name for constraint in self._constraints]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConstraintError(f"More than one constraint is named {duplicates}.")

    @classmethod
    def from_specs(cls, specs: Sequence[ConstraintSpec], catalog: ResponseCatalog | None = None) -> CertificationBasis:
        """Build the basis a case file's ``constraints`` list describes."""
        catalog = catalog if catalog is not None else ResponseCatalog()
        return cls([Constraint(spec, catalog) for spec in specs])

    def __len__(self) -> int:
        """Return the number of constraints."""
        return len(self._constraints)

    def __iter__(self) -> Iterator[Constraint]:
        """Iterate the constraints."""
        return iter(self._constraints)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """The constraints, in the order the case file states them."""
        return self._constraints

    @property
    def traceable(self) -> tuple[Constraint, ...]:
        """The constraints that name both a regulation and a source."""
        return tuple(constraint for constraint in self._constraints if constraint.is_traceable)

    def register(self, model: om.Group) -> None:
        """Declare every constraint on the model, before setup."""
        for constraint in self._constraints:
            constraint.register(model)

    def evaluate(self, box: Any) -> list[ConstraintResult]:
        """Evaluate every constraint against a converged design."""
        return [constraint.evaluate(box) for constraint in self._constraints]

    def traceability_matrix(self, box: Any) -> str:
        """Return the constraints, their provenance and their margins, as a table."""
        if not self._constraints:
            return "No constraints were declared; nothing was constrained."

        results = self.evaluate(box)
        titles = max(len(result.constraint.title) for result in results)
        regulations = max(max((len(result.constraint.regulation) for result in results), default=0), len("regulation"))
        bounds = max(max(len(r.constraint.spec.bounds.describe()) for r in results), len("bound"))

        header = (
            f"{'regulation':<{regulations}s}  {'constraint':<{titles}s}  {'value':>13s} "
            f"{'bound':>{bounds}s} {'margin':>13s}  {'units':<6s} status"
        )
        lines = [header, "-" * len(header)]
        for result in results:
            constraint = result.constraint
            lines.append(
                f"{constraint.regulation or '-':<{regulations}s}  {constraint.title:<{titles}s}  "
                f"{result.value:13.4f} {constraint.spec.bounds.describe():>{bounds}s} "
                f"{result.margin:13.4f}  {(constraint.units or '-'):<6.6s} {result.status}"
            )

        traceable = self.traceable
        if traceable:
            lines += ["", "Where each limit came from", "--------------------------"]
            for constraint in traceable:
                lines.append(f"  {constraint.name} ({constraint.regulation}): {constraint.source}")

        untraceable = [c.name for c in self._constraints if not c.is_traceable]
        if untraceable:
            lines += [
                "",
                f"Design constraints with no stated regulation or source: {', '.join(untraceable)}.",
                "These bound the design; they are not certification evidence.",
            ]

        violated = [result for result in results if not result.satisfied]
        active = [result for result in results if result.active]
        lines += [
            "",
            f"{len(results) - len(violated)} of {len(results)} constraints met, "
            f"{len(active)} active, {len(violated)} violated.",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation naming the constraint count."""
        return f"CertificationBasis({len(self._constraints)} constraints)"
