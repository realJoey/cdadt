"""The weights discipline: payload and the cabin, and the weight rollup the box closes on."""

from __future__ import annotations

from typing import ClassVar

from cdadt.disciplines.base import Discipline
from cdadt.parameters import Response

__all__ = ["Weights"]


class Weights(Discipline):
    """Payload, occupancy and cabin pressure, and the three weights that close the sizing loop.

    Owns the ``ac|weights|`` parameters and the cabin definition -- passenger capacity, flight
    deck and cabin crew, cabin pressure. Those are weight inputs rather than geometry: inside
    the black box they drive furnishings, oxygen, air conditioning and pressurization, and the
    pressurized cabin volume that the fuselage weight correlation reads.

    Reports the rollup the whole model exists to close:

    .. math::

       \\mathrm{MTOW} = \\mathrm{OEW}(\\mathrm{MTOW}, \\text{geometry})
                        + W_\\mathrm{payload}
                        + W_\\mathrm{fuel}(\\mathrm{MTOW}, \\text{mission})

    A heavier aircraft burns more fuel and more fuel makes it heavier. The black box drives that
    residual to zero with its own Newton solver, at the same time as the mission's implicit
    states -- phase durations, throttle settings and the decision speed. cdadt supplies the
    starting guess and reads the answer; the closure itself is the box's.

    Maximum landing weight is reported because it sizes the landing gear, not because it was
    chosen: the box takes it as 80% of maximum takeoff weight. That is an assumption of the box,
    not of cdadt, and is recorded in :doc:`/blackbox`.
    """

    discipline_name: ClassVar[str] = "weights"
    description: ClassVar[str] = "Payload, occupancy and cabin, and the closed weight rollup"

    owned_patterns: ClassVar[tuple[str, ...]] = (
        "ac|weights|*",
        "ac|num_*",
        "ac|cabin_pressure",
    )

    reported: ClassVar[tuple[Response, ...]] = (
        Response("MTOW", "ac|weights|MTOW", "kg", "Maximum takeoff weight, converged by the box"),
        Response("OEW", "ac|weights|OEW", "kg", "Operating empty weight, from the box's weight buildup"),
        Response("MLW", "ac|weights|MLW", "kg", "Maximum landing weight, taken by the box as 0.8 x MTOW"),
        Response("payload", "ac|weights|W_payload", "kg", "Design payload"),
        Response("furnishings_weight", "empty_weight.W_furnishings", "kg", "Furnishings", optional=True),
        Response("avionics_weight", "empty_weight.W_avionics", "kg", "Avionics", optional=True),
        Response("electrical_weight", "empty_weight.W_electrical", "kg", "Electrical system", optional=True),
        Response("apu_weight", "empty_weight.W_APU", "kg", "Auxiliary power unit", optional=True),
        Response("oxygen_weight", "empty_weight.W_oxygen", "kg", "Oxygen system", optional=True),
        Response(
            "environmental_weight",
            "empty_weight.W_ac_pressurize_antiice",
            "kg",
            "Air conditioning, pressurization and anti-ice",
            optional=True,
        ),
        Response(
            "flight_controls_weight",
            "empty_weight.W_flight_controls",
            "kg",
            "Flight control system",
            optional=True,
        ),
        Response(
            "pressurized_volume",
            "empty_weight.cabin_volume.V_pressurized",
            "m**3",
            "Pressurized cabin volume, computed by the box",
            optional=True,
        ),
    )
