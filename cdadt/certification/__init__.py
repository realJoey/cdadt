"""The certification basis: regulations as first-class modeling objects.

:mod:`cdadt.certification.basis`
    :class:`~cdadt.certification.basis.Requirement`, the abstraction, and
    :class:`~cdadt.certification.basis.CertificationBasis`, which holds the active set and
    emits the traceability matrix.

``cdadt.certification.requirements``
    The requirements themselves: balanced field length, one-engine-inoperative climb
    gradient, landing field length, approach speed, and throttle margin.

``cdadt.certification.components``
    Components cdadt authors itself, for conditions the mission does not model. Everything
    here carries analytic partials verified against complex step.

Every limit is configured, never defaulted. A field length or a climb gradient comes from a
regulation *and* an operating case -- a runway, an altitude, a temperature -- so a limit
built into the code would apply itself to aircraft nobody chose it for.
"""

from cdadt.certification.basis import CertificationBasis, Requirement, RequirementResult, Sense
from cdadt.certification.requirements import (
    ApproachSpeedLimit,
    BalancedFieldLength,
    EngineOutClimbGradient,
    LandingFieldLengthLimit,
    ThrottleMargin,
)

__all__ = [
    "ApproachSpeedLimit",
    "BalancedFieldLength",
    "CertificationBasis",
    "EngineOutClimbGradient",
    "LandingFieldLengthLimit",
    "Requirement",
    "RequirementResult",
    "Sense",
    "ThrottleMargin",
]
