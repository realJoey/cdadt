"""Landing performance components.

OpenConcept models no landing. These components are cdadt's own, so unlike the wrapped
providers they carry physics that has to be justified and verified in its own right: every
partial derivative here is analytic and checked against complex step, and every constant is
read from configuration with a recorded source rather than written into the code.

The method is the standard energy/deceleration decomposition used in conceptual design: the
landing distance from the 50 ft screen height is an airborne segment flown at the approach
speed, plus a ground segment in which the aircraft's kinetic energy is dissipated at a mean
deceleration. Both the airborne distance and the mean deceleration are configured, because
both depend on the aircraft's braking system, the runway, and the procedure flown -- none of
which a formula can infer.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

__all__ = ["ApproachSpeed", "LandingFieldLength"]


class ApproachSpeed(om.ExplicitComponent):
    """Reference landing approach speed from the stall speed in landing configuration.

    Implements ``V_REF = k V_SR0``. 14 CFR 25.125(a)(2) requires the speed at the 50 ft
    screen height to be at least 1.23 V\\ :sub:`SR0`, the reference stall speed in landing
    configuration.

    The factor is an input rather than a built-in 1.23. The regulation sets a floor, not a
    value: an operator or a manufacturer may fly a higher approach speed, and a design sized
    to exactly the floor has no margin against the procedure actually flown.

    Notes
    -----
    The ``num_nodes`` option sets how many points are evaluated; default 1.

    Inputs are ``Vstall_land`` (the reference stall speed in landing configuration) and
    ``approach_speed_factor``. The output is ``V_ref``.
    """

    def initialize(self):
        """Declare the node count."""
        self.options.declare("num_nodes", default=1, types=int, desc="Number of points to evaluate")

    def setup(self):
        """Declare inputs, outputs, and the analytic partial structure."""
        nn = self.options["num_nodes"]
        self.add_input("Vstall_land", shape=(nn,), units="m/s", desc="Reference stall speed, landing config")
        self.add_input("approach_speed_factor", shape=(nn,), desc="V_REF / V_SR0; 25.125(a)(2) requires >= 1.23")
        self.add_output("V_ref", shape=(nn,), units="m/s", desc="Reference approach speed at the 50 ft height")

        rows = np.arange(nn)
        self.declare_partials("V_ref", "Vstall_land", rows=rows, cols=rows)
        self.declare_partials("V_ref", "approach_speed_factor", rows=rows, cols=rows)

    def compute(self, inputs, outputs):
        """Compute the reference approach speed."""
        outputs["V_ref"] = inputs["approach_speed_factor"] * inputs["Vstall_land"]

    def compute_partials(self, inputs, J):
        """Compute the analytic partials of the approach speed."""
        J["V_ref", "Vstall_land"] = inputs["approach_speed_factor"]
        J["V_ref", "approach_speed_factor"] = inputs["Vstall_land"]


class LandingFieldLength(om.ExplicitComponent):
    """Landing field length from the approach speed, by the energy/deceleration method.

    The landing distance from the 50 ft screen height is

    .. math::

       s_\\text{landing} = s_\\text{air} + \\frac{V_\\text{ref}^2}{2 \\bar{a}}

    where :math:`s_\\text{air}` is the airborne distance flown from the screen height to
    touchdown and :math:`\\bar{a}` is the mean deceleration achieved on the ground. The
    demonstrated landing distance is then divided by a dispatch factor to give the field
    length required:

    .. math::

       s_\\text{field} = \\frac{s_\\text{landing}}{f_\\text{dispatch}}

    14 CFR 121.195(b) requires a turbine-powered transport to be dispatched only to an
    airport where it can land within 60% of the effective runway length, which makes
    :math:`f_\\text{dispatch} = 0.6` for that rule. It is an input because the applicable
    factor depends on the operating rule and the runway condition -- wet runway dispatch
    under 121.195(d) is a different number, and a design sized to the dry factor is not
    sized for the wet case.

    Notes
    -----
    The ``num_nodes`` option sets how many points are evaluated; default 1.

    Inputs are ``V_ref``, ``airborne_distance``, ``mean_deceleration`` and
    ``dispatch_factor``. Outputs are ``landing_distance`` (the demonstrated distance from
    50 ft) and ``landing_field_length`` (the factored distance).

    The method is deliberately transparent rather than an empirical fit. Every quantity in
    it is one an engineer can defend or measure: how far the aircraft floats, how hard it
    decelerates, and which operating rule applies. An empirical correlation would hide all
    three inside a single coefficient calibrated on aircraft that are not the one being
    designed.
    """

    def initialize(self):
        """Declare the node count."""
        self.options.declare("num_nodes", default=1, types=int, desc="Number of points to evaluate")

    def setup(self):
        """Declare inputs, outputs, and the analytic partial structure."""
        nn = self.options["num_nodes"]
        self.add_input("V_ref", shape=(nn,), units="m/s", desc="Reference approach speed")
        self.add_input("airborne_distance", shape=(nn,), units="m", desc="Distance from 50 ft to touchdown")
        self.add_input("mean_deceleration", shape=(nn,), units="m/s**2", desc="Mean deceleration on the ground")
        self.add_input("dispatch_factor", shape=(nn,), desc="Fraction of runway usable, e.g. 0.6 per 121.195(b)")

        self.add_output("landing_distance", shape=(nn,), units="m", desc="Demonstrated landing distance from 50 ft")
        self.add_output("landing_field_length", shape=(nn,), units="m", desc="Factored landing field length")

        rows = np.arange(nn)
        for output in ("landing_distance", "landing_field_length"):
            for wrt in ("V_ref", "airborne_distance", "mean_deceleration"):
                self.declare_partials(output, wrt, rows=rows, cols=rows)
        self.declare_partials("landing_field_length", "dispatch_factor", rows=rows, cols=rows)

    def compute(self, inputs, outputs):
        """Compute the demonstrated and factored landing distances."""
        ground_roll = inputs["V_ref"] ** 2 / (2.0 * inputs["mean_deceleration"])
        outputs["landing_distance"] = inputs["airborne_distance"] + ground_roll
        outputs["landing_field_length"] = outputs["landing_distance"] / inputs["dispatch_factor"]

    def compute_partials(self, inputs, J):
        """Compute the analytic partials of both distances."""
        v_ref = inputs["V_ref"]
        decel = inputs["mean_deceleration"]
        factor = inputs["dispatch_factor"]
        ones = np.ones_like(v_ref)

        d_ground_d_vref = v_ref / decel
        d_ground_d_decel = -(v_ref**2) / (2.0 * decel**2)

        J["landing_distance", "V_ref"] = d_ground_d_vref
        J["landing_distance", "airborne_distance"] = ones
        J["landing_distance", "mean_deceleration"] = d_ground_d_decel

        J["landing_field_length", "V_ref"] = d_ground_d_vref / factor
        J["landing_field_length", "airborne_distance"] = ones / factor
        J["landing_field_length", "mean_deceleration"] = d_ground_d_decel / factor

        landing_distance = inputs["airborne_distance"] + v_ref**2 / (2.0 * decel)
        J["landing_field_length", "dispatch_factor"] = -landing_distance / factor**2
