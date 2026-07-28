"""The structures discipline: the load-carrying airframe, reported component by component."""

from __future__ import annotations

from typing import ClassVar

from cdadt.disciplines.base import Discipline, DisciplineError
from cdadt.parameters import Parameter, Response

__all__ = ["Structures"]


class Structures(Discipline):
    """Mass of the load-carrying airframe, broken down by component.

    **This discipline owns no inputs, and that is the honest answer.** Primary structure is
    sized inside the black box by empirical correlations that read geometry, maximum takeoff
    weight and maximum landing weight -- all of which are owned by
    :class:`~cdadt.disciplines.geometry.Geometry`,
    :class:`~cdadt.disciplines.stability.Stability` and
    :class:`~cdadt.disciplines.weights.Weights`, or computed by the box itself. There is no
    structural parameter left for cdadt to set: no material, no load factor, no spar layout.
    Inventing one here so that the class had state would be a parameter nothing reads.

    What it does own is the *reporting* of structure. A structural mass that cannot be broken
    down cannot be argued with, and a design report that gives only operating empty weight
    hides where the weight went. This class names the seven component masses the box publishes,
    where each lives, and in what units to read it.

    All of them are marked optional because they exist in the box only as long as the box's
    empty-weight model publishes them. A different sizing model, named in the case file, may
    not; the run then reports them as unavailable rather than failing or, worse, quietly
    dropping them.

    Notes
    -----
    The ultimate load factor, the structural allowance and every other constant in those
    correlations are options of components inside the box. They are not settable from cdadt and
    are documented as fixed by the box in :doc:`/blackbox`.
    """

    discipline_name: ClassVar[str] = "structures"
    description: ClassVar[str] = "Load-carrying airframe mass, component by component"

    #: Structure has no settable parameter of its own; see the class docstring.
    owned_patterns: ClassVar[tuple[str, ...]] = ()

    reported: ClassVar[tuple[Response, ...]] = (
        Response(
            "structure_weight",
            "empty_weight.W_structure",
            "kg",
            "Total primary structure mass, allowance included",
            optional=True,
        ),
        Response("wing_weight", "empty_weight.W_wing", "kg", "Wing structural mass", optional=True),
        Response("hstab_weight", "empty_weight.W_hstab", "kg", "Horizontal stabilizer mass", optional=True),
        Response("vstab_weight", "empty_weight.W_vstab", "kg", "Vertical stabilizer mass", optional=True),
        Response("fuselage_weight", "empty_weight.W_fuselage", "kg", "Fuselage structural mass", optional=True),
        Response("main_gear_weight", "empty_weight.W_mlg", "kg", "Main landing gear mass", optional=True),
        Response("nose_gear_weight", "empty_weight.W_nlg", "kg", "Nose landing gear mass", optional=True),
        Response("nacelle_weight", "empty_weight.W_nacelle", "kg", "Nacelle mass", optional=True),
    )

    def add(self, parameter: Parameter) -> None:
        """Reject every parameter, with an explanation.

        Raises
        ------
        DisciplineError
            Always. The base class would raise anyway, since nothing matches an empty ownership
            pattern, but the generic message would read as a gap in the ownership map rather
            than as a deliberate property of this domain.
        """
        raise DisciplineError(
            f"The structures discipline sets nothing: '{parameter.name}' cannot belong to it. Primary "
            f"structure is sized inside the black box from geometry and weights, so its inputs are "
            f"owned by the geometry, stability and weights disciplines."
        )
