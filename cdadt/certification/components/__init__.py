"""Components cdadt authors itself, for conditions the mission does not model.

Unlike the wrapped providers, everything here carries physics cdadt is responsible for. So
everything here also carries analytic partial derivatives verified against complex step, and
reads its constants from configuration rather than embedding them.
"""

from cdadt.certification.components.landing import ApproachSpeed, LandingFieldLength

__all__ = ["ApproachSpeed", "LandingFieldLength"]
