"""The certification requirements cdadt enforces."""

from cdadt.certification.requirements.landing import (
    LANDING_GROUP,
    ApproachSpeedLimit,
    LandingFieldLengthLimit,
)
from cdadt.certification.requirements.takeoff import BalancedFieldLength, EngineOutClimbGradient
from cdadt.certification.requirements.thrust_margin import ThrottleMargin

__all__ = [
    "LANDING_GROUP",
    "ApproachSpeedLimit",
    "BalancedFieldLength",
    "EngineOutClimbGradient",
    "LandingFieldLengthLimit",
    "ThrottleMargin",
]
