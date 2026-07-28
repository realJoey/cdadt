"""The performance discipline: the mission flown, and everything the mission produces."""

from __future__ import annotations

from typing import Any, ClassVar

from cdadt.disciplines.base import Discipline, DisciplineError
from cdadt.mission import ContinuationLadder, InitialConditions
from cdadt.parameters import Parameter, Response

__all__ = ["Performance"]


class Performance(Discipline):
    """The mission the aircraft is sized against, and the results of flying it.

    The one discipline whose encapsulated state is not a bag of scalars. What performance owns
    is the pair OpenConcept's run scripts write by hand: the initial conditions
    (``set_values(prob, num_nodes)``) and the continuation ladder that converges the hard ones
    (the two ``run_model()`` calls inside ``set_mission_profile``).

    What it reports is everything the mission produces, and it is what the certification
    constraints are written against: balanced field length and its abort distance, the decision
    and takeoff safety speeds, the engine-out climb gradient, block fuel and fuel with reserves,
    the range actually flown, and the throttle history of every phase.

    Parameters
    ----------
    conditions : InitialConditions
        Everything written into the box before it is converged.
    ladder : ContinuationLadder, optional
        The rungs walked first. Empty attempts the design mission directly.
    """

    discipline_name: ClassVar[str] = "performance"
    description: ClassVar[str] = "The mission flown, and the field, climb and fuel results it produces"

    #: Performance owns the mission-level values. They live in the initial conditions rather
    #: than as loose parameters, so :meth:`add` refuses them; the pattern is declared so that
    #: the ownership check over the box's settable variables comes out total.
    owned_patterns: ClassVar[tuple[str, ...]] = ("mission.*",)

    reported: ClassVar[tuple[Response, ...]] = (
        Response(
            "block_fuel",
            "mission.descent.fuel_burn_integ.fuel_burn_final",
            "kg",
            "Fuel burned by the end of the design mission, reserves excluded",
        ),
        Response(
            "total_fuel",
            "mission.loiter.fuel_burn_integ.fuel_burn_final",
            "kg",
            "Fuel burned by the end of loiter: the block fuel plus the reserve mission",
        ),
        Response(
            "takeoff_field_length",
            "mission.bfl.distance_continue",
            "ft",
            "Balanced field length: distance to 35 ft continuing after failure at V1",
        ),
        Response(
            "abort_distance",
            "mission.bfl.distance_abort",
            "ft",
            "Distance to a full stop rejecting the takeoff at V1; equals the field length at convergence",
        ),
        Response("V1", "mission.takeoff|v1", "kn", "Decision speed, solved implicitly by the box"),
        Response("V2", "mission.engineoutclimb.takeoff|v2", "kn", "Takeoff safety speed"),
        Response(
            "engine_out_climb_gradient",
            "mission.engineoutclimb.gamma",
            "rad",
            "Climb angle at V2 with one engine inoperative",
        ),
        Response(
            "mission_range_flown",
            "mission.descent.ode_integ_phase.range_final",
            "nmi",
            "Ground distance covered by the end of descent",
        ),
        Response(
            "reserve_range_flown",
            "mission.resrange.reserverange",
            "nmi",
            "Ground distance covered by the reserve diversion",
            optional=True,
        ),
        Response("climb_throttle", "mission.climb.throttle", None, "Throttle at every climb node"),
        Response("cruise_throttle", "mission.cruise.throttle", None, "Throttle at every cruise node"),
        Response("descent_throttle", "mission.descent.throttle", None, "Throttle at every descent node"),
        Response("climb_duration", "mission.climb.duration", "min", "Climb duration, solved to reach cruise altitude"),
        Response("cruise_duration", "mission.cruise.duration", "min", "Cruise duration, solved to reach the range"),
        Response("descent_duration", "mission.descent.duration", "min", "Descent duration, solved to reach the ground"),
        Response("loiter_duration", "mission.loiter.duration", "min", "Loiter duration, as specified", optional=True),
    )

    def __init__(self, conditions: InitialConditions, ladder: ContinuationLadder | None = None) -> None:
        """Hold the initial conditions and the ladder that converges them."""
        super().__init__()
        self._conditions = conditions
        self._ladder = ladder if ladder is not None else ContinuationLadder()

    @property
    def conditions(self) -> InitialConditions:
        """Everything written into the box before it is converged."""
        return self._conditions

    @property
    def ladder(self) -> ContinuationLadder:
        """The rungs walked before the design mission."""
        return self._ladder

    def add(self, parameter: Parameter) -> None:
        """Reject loose parameters, with an explanation.

        Raises
        ------
        DisciplineError
            Always. Mission-level values belong in ``initial_conditions``, alongside the
            schedules and the ladder that make them reachable.
        """
        raise DisciplineError(
            f"'{parameter.name}' cannot be added to the performance discipline as a loose parameter. "
            f"Mission-level values belong under 'initial_conditions' in the case file."
        )

    def apply(self, box: Any) -> None:
        """Write the design conditions into the black box, without converging it."""
        self._conditions.apply(box)

    def converge(self, box: Any, verbose: bool = False) -> None:
        """Walk the continuation ladder and converge the design mission."""
        self._ladder.converge(box, self._conditions, verbose=verbose)

    def __repr__(self) -> str:
        """Return a representation naming the conditions and the ladder."""
        return f"Performance({self._conditions!r}, {self._ladder!r})"
