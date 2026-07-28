"""The performance discipline: the mission flown, and everything the mission produces."""

from __future__ import annotations

from typing import Any, ClassVar

from cdadt.disciplines.base import Discipline, DisciplineError
from cdadt.mission import MissionProfile
from cdadt.parameters import Parameter, Response

__all__ = ["Performance"]


class Performance(Discipline):
    """The mission the aircraft is sized against, and the results of flying it.

    The one discipline whose encapsulated state is not a bag of scalars. What performance owns
    is a :class:`~cdadt.mission.MissionProfile`: the design range, the cruise altitude, the
    reserve mission, the loiter, the speed and vertical-speed schedule flown in each of the
    seven steady-flight phases, and the continuation ladder that converges them. Those are
    design inputs in exactly the sense the geometry parameters are -- change the range and you
    change the aircraft -- but they are a structured object rather than a list of numbers, so
    they are held as one.

    What it reports is everything the mission produces, and it is the discipline the
    certification requirements are written against: balanced field length and its abort
    distance, the decision and takeoff safety speeds, the engine-out climb gradient, block fuel
    and fuel with reserves, the range actually flown, and the throttle history of every phase.

    Parameters
    ----------
    profile : MissionProfile
        The mission and its continuation ladder.
    num_nodes : int
        Analysis points per phase, needed to resample the schedules onto the box's grid.

    Notes
    -----
    Field length here is OpenConcept's balanced field length: the distance at which continuing
    the takeoff after an engine failure at V\\ :sub:`1` and rejecting it cover the same ground.
    The box solves V\\ :sub:`1` implicitly to make that true, so a converged run has
    ``takeoff_field_length == abort_distance``. A run where they differ has not converged, and
    the validation suite asserts it.
    """

    discipline_name: ClassVar[str] = "performance"
    description: ClassVar[str] = "The mission flown, and the field, climb and fuel results it produces"

    #: Performance owns the mission-level parameters. They are held inside the profile rather
    #: than as loose parameters, so :meth:`add` refuses them; the pattern is declared so that
    #: the ownership check over the black box's settable variables comes out total.
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

    def __init__(self, profile: MissionProfile, num_nodes: int) -> None:
        """Hold the mission profile and the grid it will be resampled onto."""
        super().__init__()
        self._profile = profile
        self._num_nodes = int(num_nodes)

    @property
    def profile(self) -> MissionProfile:
        """The mission profile this discipline owns."""
        return self._profile

    @property
    def num_nodes(self) -> int:
        """Analysis points per phase."""
        return self._num_nodes

    def add(self, parameter: Parameter) -> None:
        """Reject loose parameters, with an explanation.

        Raises
        ------
        DisciplineError
            Always. Mission-level values belong in the profile, where they sit alongside the
            schedules and the continuation ladder that make them reachable.
        """
        raise DisciplineError(
            f"'{parameter.name}' cannot be added to the performance discipline as a loose parameter. "
            f"Mission-level values belong in the mission profile, under 'mission.parameters' in the case file."
        )

    def apply(self, box: Any) -> None:
        """Write the design mission into the black box, without converging it."""
        self._profile.apply(box, self._num_nodes)

    def converge(self, box: Any, verbose: bool = False) -> None:
        """Walk the continuation ladder and converge the design mission.

        Parameters
        ----------
        box : OpenConceptSizingBox
            The black box, already built.
        verbose : bool, optional
            Print each continuation step as it runs. Default ``False``.
        """
        self._profile.converge(box, self._num_nodes, verbose=verbose)

    def __repr__(self) -> str:
        """Return a representation naming the profile and the node count."""
        return f"Performance({self._profile!r}, num_nodes={self._num_nodes})"
