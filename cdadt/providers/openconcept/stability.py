"""Empennage sizing wrapping OpenConcept's tail volume coefficient components."""

from __future__ import annotations

import openmdao.api as om
from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing

from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

__all__ = ["TailVolumeCoefficientProvider"]


class TailVolumeCoefficientProvider(Provider):
    """Tail areas from the volume coefficient method.

    Sizes both stabilizers from a configured tail volume coefficient, the wing planform,
    and the tail moment arms supplied by the geometry discipline.

    Notes
    -----
    **Required configuration**, with no defaults:
    ``ac|geom|hstab|volume_coefficient`` and ``ac|geom|vstab|volume_coefficient``.

    OpenConcept declares these as OpenMDAO options defaulting to Raymer's Table 6.4 values
    for a jet transport (1.00 horizontal, 0.09 vertical). Those are defensible values *for
    a jet transport*, which is exactly why they are dangerous as defaults: a configuration
    that never mentioned them would still produce tail areas, and nothing in the result
    would say which class of aircraft they came from. Configuring them explicitly, with a
    ``source`` recorded alongside, puts the citation in the run report instead.
    """

    @property
    def name(self) -> str:
        """Return ``"tail_volume_coefficient"``."""
        return "tail_volume_coefficient"

    @property
    def reference(self) -> str:
        """Return the citation for this method."""
        return (
            "Raymer, Aircraft Design: A Conceptual Approach, 1992, Section 6.4, Equation 6.27 "
            "and Table 6.4, as implemented by OpenConcept's HStabVolumeCoefficientSizing and "
            "VStabVolumeCoefficientSizing."
        )

    def provides(self) -> VariableSet:
        """Return the stabilizer reference areas."""
        return VariableSet(
            [
                Variable("ac|geom|hstab|S_ref", "m**2", description="Horizontal stabilizer reference area"),
                Variable("ac|geom|vstab|S_ref", "m**2", description="Vertical stabilizer reference area"),
            ]
        )

    def requires(self) -> VariableSet:
        """Return the wing planform and tail moment arms this method reads."""
        return VariableSet(
            [
                Variable("ac|geom|wing|S_ref", "m**2", description="Wing reference area"),
                Variable("ac|geom|wing|AR", None, description="Wing aspect ratio"),
                Variable("ac|geom|wing|MAC", "m", description="Wing mean aerodynamic chord"),
                Variable("ac|geom|hstab|c4_to_wing_c4", "m", description="Horizontal tail moment arm"),
                Variable("ac|geom|vstab|c4_to_wing_c4", "m", description="Vertical tail moment arm"),
            ]
        )

    def validate_configuration(self) -> None:
        """Require both volume coefficients; OpenConcept's option defaults are not used."""
        self.config.require_all(
            [
                "ac|geom|hstab|volume_coefficient",
                "ac|geom|vstab|volume_coefficient",
            ]
        )

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add the two sizing components to ``group``.

        ``num_nodes`` and ``flight_phase`` are unused: tail area is a design quantity, which
        is why this provider backs an aircraft-scoped discipline.
        """
        group.add_subsystem(
            "hstab_area",
            HStabVolumeCoefficientSizing(C_ht=self.config.scalar("ac|geom|hstab|volume_coefficient")),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|MAC", "ac|geom|hstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|hstab|S_ref"],
        )
        group.add_subsystem(
            "vstab_area",
            VStabVolumeCoefficientSizing(C_vt=self.config.scalar("ac|geom|vstab|volume_coefficient")),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|geom|vstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|vstab|S_ref"],
        )
