"""The propulsion discipline: the engine rating and count, and the installed weights."""

from __future__ import annotations

from typing import ClassVar

from cdadt.disciplines.base import Discipline
from cdadt.parameters import Response

__all__ = ["Propulsion"]


class Propulsion(Discipline):
    """Engine rating and engine count, and the propulsion installation weights.

    Owns the two ``ac|propulsion|`` parameters. The rating is the sea-level static thrust the
    engine deck inside the black box is scaled to -- OpenConcept calls this a *rubberized*
    engine, meaning thrust and fuel flow are scaled from a fixed CFM56 deck rather than
    recomputed by a cycle analysis. Engine count multiplies both, and is what makes the
    engine-out cases meaningful: the box multiplies by the number of *active* engines, which is
    one fewer in the rejected-takeoff and engine-out climb conditions.

    Thrust, fuel flow and throttle at every node are computed inside the box and belong to the
    mission, so they are reported by
    :class:`~cdadt.disciplines.performance.Performance`. What this discipline reports is the
    weight of the propulsion installation, which is what the rating and count size.

    Notes
    -----
    Because the engine is a scaled deck, a very large change in rating extrapolates the
    surrogate rather than redesigning an engine. :doc:`/validation` records that limit.
    """

    discipline_name: ClassVar[str] = "propulsion"
    description: ClassVar[str] = "Engine rating and count, and the propulsion installation weights"

    owned_patterns: ClassVar[tuple[str, ...]] = ("ac|propulsion|*",)

    reported: ClassVar[tuple[Response, ...]] = (
        Response(
            "engine_weight",
            "empty_weight.single_engine.W_engine",
            "kg",
            "Dry weight of one engine, from the box's weight buildup",
            optional=True,
        ),
        Response(
            "engines_weight",
            "empty_weight.W_engines",
            "kg",
            "Installed weight of all engines",
            optional=True,
        ),
        Response(
            "thrust_reverser_weight",
            "empty_weight.W_thrust_rev",
            "kg",
            "Thrust reverser weight",
            optional=True,
        ),
        Response(
            "engine_controls_weight",
            "empty_weight.W_eng_control",
            "kg",
            "Engine control system weight",
            optional=True,
        ),
        Response(
            "engine_starter_weight",
            "empty_weight.W_eng_start",
            "kg",
            "Engine starting system weight",
            optional=True,
        ),
        Response(
            "fuel_system_weight",
            "empty_weight.W_fuelsystem",
            "kg",
            "Fuel system weight, sized by the maximum fuel load",
            optional=True,
        ),
    )
