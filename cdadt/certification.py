"""The certification basis: regulations as first-class modeling objects.

This is what makes cdadt certification-*driven* rather than certification-*checked*. A
:class:`Requirement` is not a constraint expression buried in a run script. It is an object
that knows the regulation it enforces, the limit that regulation imposes and where that limit
came from, the model quantity it reads, any analysis it has to contribute to evaluate itself,
and how to report its own margin afterwards.

The artifact is the traceability matrix: after any run, every regulation appears alongside the
value achieved, the limit, the margin, and whether it was binding. A design that merely
converged does not tell you which requirements shaped it; this does.

No limit is defaulted. A field length or a required climb gradient comes from a regulation
*and* an operating case -- a runway, an altitude, a fleet category -- so it is a constructor
argument with no fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import openmdao.api as om
from openconcept.aerodynamics import FlapCLmax, StallSpeed

__all__ = [
    "ApproachSpeed",
    "BalancedFieldLength",
    "CertificationBasis",
    "EngineOutClimbGradient",
    "Requirement",
    "RequirementResult",
    "Sense",
    "ThrottleMargin",
]


class Sense:
    """Which side of the limit satisfies a requirement."""

    #: The quantity must not exceed the limit, e.g. a field length.
    UPPER = "upper"

    #: The quantity must not fall below the limit, e.g. a climb gradient.
    LOWER = "lower"


@dataclass(frozen=True)
class RequirementResult:
    """One row of the traceability matrix.

    Parameters
    ----------
    requirement : Requirement
        The requirement this result belongs to.
    value : float
        The value the design achieved. For a vector quantity it is the worst element, since a
        requirement holds only if it holds everywhere.
    limit : float
        The limit enforced.
    units : str or None
        Units of both.
    margin : float
        Signed margin: ``limit - value`` for an upper limit, ``value - limit`` for a lower one.
        Positive means satisfied; zero means exactly binding.
    """

    #: Relative margin within which a requirement counts as binding rather than violated. A
    #: converged optimizer lands an active constraint a hair either side of its bound, so
    #: testing satisfaction by exact sign would report it as violated by 1e-14 of a knot. This
    #: is a reporting tolerance, not a relaxation of the constraint the optimizer enforced.
    ACTIVE_TOLERANCE: ClassVar[float] = 1.0e-6

    requirement: Requirement
    value: float
    limit: float
    units: str | None
    margin: float

    @property
    def satisfied(self) -> bool:
        """Return whether the design meets the requirement."""
        return self.margin >= 0.0

    @property
    def active(self) -> bool:
        """Return whether the requirement is binding, within tolerance.

        An active requirement is the interesting outcome of an optimization: it is a
        requirement that shaped the design.
        """
        return abs(self.margin) <= self.ACTIVE_TOLERANCE * max(abs(self.limit), 1.0)

    @property
    def status(self) -> str:
        """Return ``"ACTIVE"``, ``"MET"`` or ``"NOT MET"``."""
        if self.active:
            return "ACTIVE"
        return "MET" if self.satisfied else "NOT MET"

    @property
    def relative_margin(self) -> float:
        """Return the margin as a fraction of the limit.

        Margins in feet and margins in radians cannot be compared; their relative margins can,
        which is what lets the traceability matrix be sorted most-binding first.
        """
        return float("inf") if self.limit == 0.0 else self.margin / abs(self.limit)


class Requirement(ABC):
    """One certification requirement, as an object.

    Parameters
    ----------
    limit : float
        The limit, in :attr:`units`.
    source : str
        Where the limit came from -- the runway, the operating case, the fleet category.
        Required, because a limit without provenance is indistinguishable from a guess.
    """

    def __init__(self, limit: float, source: str) -> None:
        self.limit = float(limit)
        self.source = source

    # -- identity -----------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a short identifier, used as a subsystem name and a report key."""

    @property
    @abstractmethod
    def regulation(self) -> str:
        """Return the citation, e.g. ``"14 CFR 25.113"``."""

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
        """Return the units the constrained quantity and its limit are in."""

    @property
    @abstractmethod
    def path(self) -> str:
        """Return the problem path of the quantity this requirement constrains."""

    # -- model contribution -------------------------------------------------------------

    def build(self, model: om.Group) -> None:  # noqa: B027 -- optional hook
        """Add any analysis this requirement needs to evaluate itself.

        The default adds nothing: most requirements constrain a quantity the mission already
        produces. Requirements covering a condition the mission does not fly override this.
        """

    def register(self, model: om.Group) -> None:
        """Register this requirement's constraint, scaled to order one.

        The scaling is ``1 / |limit|``, and it is not cosmetic. A certification basis spans a
        field length of thousands of feet, an approach speed of a hundred-odd knots and a
        climb gradient of a few hundredths of a radian. Unscaled, an optimizer treats the
        field length as five orders of magnitude more important than the gradient and the
        gradient as satisfied to within noise.
        """
        scaler = 1.0 / abs(self.limit) if self.limit != 0.0 else 1.0
        model.add_constraint(self.path, units=self.units, scaler=scaler, **{self.sense: self.limit})

    def evaluate(self, problem: om.Problem) -> RequirementResult:
        """Read the achieved value from a converged problem and report the margin."""
        values = np.asarray(problem.get_val(self.path, units=self.units)).reshape(-1)
        if self.sense == Sense.UPPER:
            value = float(values.max())
            margin = self.limit - value
        else:
            value = float(values.min())
            margin = value - self.limit
        return RequirementResult(self, value=value, limit=self.limit, units=self.units, margin=margin)

    def __repr__(self) -> str:
        """Return a representation naming the regulation and the limit."""
        return f"{type(self).__name__}({self.regulation}, limit={self.limit} {self.units or ''})"


# ======================================================================================
# Requirements that constrain a quantity the mission already produces
# ======================================================================================


class BalancedFieldLength(Requirement):
    """14 CFR 25.113: the takeoff distance must fit the field available.

    §25.113 defines takeoff distance as the greater of the distance to 35 ft with the critical
    engine failed at V\\ :sub:`EF` and 115% of the all-engines distance. The *balanced* field
    is the condition where continuing and stopping cover the same distance, which is what
    V\\ :sub:`1` is chosen to achieve -- and OpenConcept already solves V\\ :sub:`1` implicitly
    to make them match.

    This requirement therefore adds no physics. What it adds is that the limit is declared:
    which runway the aircraft is being sized for is stated, constrained, and reported, rather
    than being a number in a run script.
    """

    @property
    def name(self) -> str:
        """Return ``"far25_113_field_length"``."""
        return "far25_113_field_length"

    @property
    def regulation(self) -> str:
        """Return the citation."""
        return "14 CFR 25.113"

    @property
    def title(self) -> str:
        """Return the one-line statement."""
        return "Takeoff distance within field available"

    @property
    def sense(self) -> str:
        """Return :attr:`Sense.UPPER`."""
        return Sense.UPPER

    @property
    def units(self) -> str:
        """Return ``"ft"``."""
        return "ft"

    @property
    def path(self) -> str:
        """Return the path of the balanced field length."""
        return "mission.bfl.distance_continue"


class EngineOutClimbGradient(Requirement):
    """14 CFR 25.121(b): second-segment climb gradient with the critical engine inoperative.

    §25.121(b) requires a steady gradient of climb at V\\ :sub:`2`, critical engine
    inoperative, landing gear retracted and takeoff flaps set, of at least 2.4% for two-engine
    aeroplanes, 2.7% for three and 3.0% for four. Which applies is a property of the aircraft,
    so the required gradient is given rather than inferred.

    Parameters
    ----------
    limit : float
        Required gradient, dimensionless rise over run -- 0.024 for a twin.
    source : str
        Provenance of the limit.

    Notes
    -----
    The regulation states a gradient, a tangent; OpenConcept reports the climb angle in
    radians. :attr:`limit` is converted with ``arctan``, which is exactly equivalent because
    ``tan`` is monotonic over the range of interest. For 2.4% the difference is under 0.1% of
    the value -- small, but real, and conflating them is how small errors get into a
    certification argument.

    The takeoff-flap part of the regulation is a property of the *model*, not of this
    requirement: OpenConcept's B738 example evaluates the engine-out climb condition with
    clean drag. Use :class:`~cdadt.model.Part25PhaseModel` to deploy takeoff flaps for it.
    """

    def __init__(self, limit: float, source: str) -> None:
        super().__init__(limit, source)
        self.gradient = self.limit
        self.limit = float(np.arctan(self.gradient))

    @property
    def name(self) -> str:
        """Return ``"far25_121b_oei_climb"``."""
        return "far25_121b_oei_climb"

    @property
    def regulation(self) -> str:
        """Return the citation."""
        return "14 CFR 25.121(b)"

    @property
    def title(self) -> str:
        """Return the one-line statement."""
        return "OEI second-segment climb gradient"

    @property
    def sense(self) -> str:
        """Return :attr:`Sense.LOWER`."""
        return Sense.LOWER

    @property
    def units(self) -> str:
        """Return ``"rad"`` -- the constrained quantity is an angle."""
        return "rad"

    @property
    def path(self) -> str:
        """Return the path of the engine-out climb angle."""
        return "mission.engineoutclimb.gamma"


class ThrottleMargin(Requirement):
    """The engine must not be asked for more than it can deliver.

    Not a regulation but an engineering limit, and it is included because without it the
    optimizer will happily shrink the engine until the mission is flown at a throttle setting
    the deck was never fitted to. It is reported as such: the traceability matrix names its
    source as design rather than as a citation.

    Parameters
    ----------
    phase : str
        Mission phase to constrain, as OpenConcept names it.
    limit : float
        Maximum throttle. 1.0 is the rated condition.
    source : str
        Provenance.
    """

    def __init__(self, phase: str, limit: float, source: str) -> None:
        super().__init__(limit, source)
        self.phase = phase

    @property
    def name(self) -> str:
        """Return ``"throttle_<phase>"``."""
        return f"throttle_{self.phase}"

    @property
    def regulation(self) -> str:
        """Return ``"design"`` -- this is an engineering limit, not a regulation."""
        return "design"

    @property
    def title(self) -> str:
        """Return the one-line statement."""
        return f"Throttle within limit in {self.phase}"

    @property
    def sense(self) -> str:
        """Return :attr:`Sense.UPPER`."""
        return Sense.UPPER

    @property
    def units(self) -> None:
        """Return ``None`` -- throttle is dimensionless."""
        return None

    @property
    def path(self) -> str:
        """Return the path of the phase's throttle vector."""
        return f"mission.{self.phase}.throttle"


# ======================================================================================
# Requirements that contribute analysis of their own
# ======================================================================================


class ApproachSpeed(Requirement):
    """Reference landing approach speed within an aerodrome category limit.

    14 CFR 25.125 defines the landing distance from a point 50 ft above the surface at an
    approach speed of at least 1.23 V\\ :sub:`SR0` (§25.125(a)(2), harmonized with §25.107(c)).
    That reference speed is what sorts an aeroplane into an approach category -- ICAO Doc 8168
    and 14 CFR 97 category C is 121-140 kn -- and the category decides which aerodromes and
    approach procedures the type may use. Constraining it is therefore an operational
    requirement with regulatory teeth, and it bites on wing area.

    OpenConcept's mission does not fly a landing, so this requirement contributes the analysis
    it needs, built from OpenConcept components only:

    .. math::

       C_{L_{max,land}} = \\mathrm{FlapCLmax}(\\delta_\\mathrm{land}),
       \\qquad
       V_\\mathrm{SR0} = \\sqrt{\\frac{2 W_\\mathrm{MLW} g}{\\rho_0 S_\\mathrm{ref} C_{L_{max,land}}}},
       \\qquad
       V_\\mathrm{ref} = k\\,V_\\mathrm{SR0}

    Parameters
    ----------
    limit : float
        Maximum reference speed, in knots.
    source : str
        Provenance of the limit -- the category and the document it comes from.
    reference_speed_factor : float, optional
        :math:`k`, the multiple of the reference stall speed. Default 1.23, the §25.125(a)(2)
        minimum. Raising it models an operator flying a larger margin.

    Notes
    -----
    Requires ``ac|aero|landing_flap_deg`` in the aircraft definition. The stall speed is an
    equivalent airspeed at sea-level density, which is the convention the category limits are
    quoted in.
    """

    def __init__(self, limit: float, source: str, reference_speed_factor: float = 1.23) -> None:
        super().__init__(limit, source)
        self.reference_speed_factor = float(reference_speed_factor)

    @property
    def name(self) -> str:
        """Return ``"approach_speed"``."""
        return "approach_speed"

    @property
    def regulation(self) -> str:
        """Return the citation."""
        return "14 CFR 25.125 / 97"

    @property
    def title(self) -> str:
        """Return the one-line statement."""
        return "Approach speed within category"

    @property
    def sense(self) -> str:
        """Return :attr:`Sense.UPPER`."""
        return Sense.UPPER

    @property
    def units(self) -> str:
        """Return ``"kn"``."""
        return "kn"

    @property
    def path(self) -> str:
        """Return the path of the reference approach speed."""
        return f"{self.name}.Vref"

    def build(self, model: om.Group) -> None:
        """Add the landing high-lift, stall-speed and reference-speed calculation."""
        group = model.add_subsystem(
            self.name,
            om.Group(),
            promotes_inputs=[
                "ac|aero|landing_flap_deg",
                "ac|aero|CLmax_cruise",
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                "ac|geom|wing|S_ref",
                "ac|weights|MLW",
            ],
        )
        group.add_subsystem(
            "landing_clmax",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|landing_flap_deg"),
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
            ],
            promotes_outputs=[("CL_max_flap", "CLmax_land")],
        )
        group.add_subsystem(
            "stall_speed",
            StallSpeed(),
            promotes_inputs=[("CLmax", "CLmax_land"), ("weight", "ac|weights|MLW"), "ac|geom|wing|S_ref"],
            promotes_outputs=["Vstall_eas"],
        )
        group.add_subsystem(
            "reference_speed",
            om.ExecComp(
                f"Vref = {self.reference_speed_factor} * Vstall_eas",
                Vref={"units": "m/s"},
                Vstall_eas={"units": "m/s"},
            ),
            promotes_inputs=["Vstall_eas"],
            promotes_outputs=["Vref"],
        )


# ======================================================================================
# The basis
# ======================================================================================


class CertificationBasis:
    """The set of requirements a design is being certified against.

    Parameters
    ----------
    requirements : sequence of Requirement
        The active requirements.

    Raises
    ------
    ValueError
        If two requirements share a name, which would make a row of the traceability matrix
        ambiguous.
    """

    def __init__(self, requirements: Sequence[Requirement]) -> None:
        self.requirements = tuple(requirements)
        names = [requirement.name for requirement in self.requirements]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Requirement names must be unique; duplicated: {duplicates}")

    def __len__(self) -> int:
        """Return the number of requirements."""
        return len(self.requirements)

    def __iter__(self):
        """Iterate the requirements."""
        return iter(self.requirements)

    def build(self, model: om.Group) -> None:
        """Add every requirement's analysis to the model."""
        for requirement in self.requirements:
            requirement.build(model)

    def register(self, model: om.Group) -> None:
        """Register every requirement's constraint."""
        for requirement in self.requirements:
            requirement.register(model)

    def evaluate(self, problem: om.Problem) -> list[RequirementResult]:
        """Evaluate every requirement against a converged problem, in declaration order."""
        return [requirement.evaluate(problem) for requirement in self.requirements]

    def traceability_matrix(self, problem: om.Problem) -> str:
        """Return the traceability matrix as a formatted table, most-binding first."""
        results = sorted(self.evaluate(problem), key=lambda result: result.relative_margin)

        header = (
            f"{'regulation':<22s} {'requirement':<36s} {'value':>12s} {'limit':>12s} "
            f"{'margin':>12s} {'units':<6s} status"
        )
        lines = [header, "-" * len(header)]
        for result in results:
            lines.append(
                f"{result.requirement.regulation:<22s} {result.requirement.title[:36]:<36s} "
                f"{result.value:12.4f} {result.limit:12.4f} {result.margin:12.4f} "
                f"{result.units or '-':<6s} {result.status}"
            )

        lines += ["", "Limit sources:"]
        seen = set()
        for requirement in self.requirements:
            key = (requirement.regulation, requirement.source)
            if key not in seen:
                seen.add(key)
                lines.append(f"  {requirement.regulation}: {requirement.source}")

        active = [result for result in results if result.active]
        unmet = [result for result in results if not result.satisfied and not result.active]
        lines += [
            "",
            f"{len(results) - len(unmet)} of {len(results)} requirements met "
            f"({len(active)} active, {len(unmet)} not met).",
        ]
        if active:
            lines.append(
                "Active requirements shaped this design: "
                + ", ".join(result.requirement.regulation for result in active)
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a representation listing the regulations."""
        return f"CertificationBasis({[r.regulation for r in self.requirements]})"
