"""Unit: disciplines own their slice of the interface, and refuse anything else."""

from __future__ import annotations

import pytest

from cdadt import (
    Aerodynamics,
    Aircraft,
    AircraftError,
    Discipline,
    DisciplineError,
    Geometry,
    Parameter,
    Performance,
    Propulsion,
    Stability,
    Structures,
    Weights,
)
from cdadt.disciplines import AIRCRAFT_DISCIPLINES


@pytest.mark.unit
def test_the_base_class_cannot_be_instantiated():
    """It declares no domain, owns nothing and reports nothing."""
    with pytest.raises(TypeError, match="base class"):
        Discipline()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("discipline", "owned", "not_owned"),
    [
        (Geometry, "ac|geom|wing|S_ref", "ac|geom|hstab|AR"),
        (Geometry, "ac|geom|nosegear|length", "ac|aero|polar|e"),
        (Aerodynamics, "ac|aero|polar|e", "ac|geom|wing|AR"),
        (Propulsion, "ac|propulsion|engine|rating", "ac|weights|W_payload"),
        (Stability, "ac|geom|vstab|AR", "ac|geom|wing|AR"),
        (Weights, "ac|weights|W_payload", "ac|geom|wing|AR"),
        (Weights, "ac|num_passengers_max", "ac|propulsion|num_engines"),
        (Weights, "ac|cabin_pressure", "ac|aero|Mach_max"),
        (Performance, "mission.mission_range", "ac|geom|wing|AR"),
    ],
)
def test_ownership_is_what_the_domain_says_it_is(discipline, owned, not_owned):
    """Each discipline claims its own domain and no other."""
    assert discipline.owns(owned)
    assert not discipline.owns(not_owned)


@pytest.mark.unit
def test_the_ownership_patterns_do_not_overlap():
    """Checked over the patterns themselves, independently of any black box."""
    disciplines = (*AIRCRAFT_DISCIPLINES, Performance)
    samples = [
        "ac|geom|wing|S_ref",
        "ac|geom|fuselage|length",
        "ac|geom|nacelle|diameter",
        "ac|geom|maingear|length",
        "ac|geom|nosegear|num_wheels",
        "ac|geom|hstab|AR",
        "ac|geom|vstab|toverc",
        "ac|aero|polar|e",
        "ac|propulsion|num_engines",
        "ac|weights|W_payload",
        "ac|num_cabin_crew",
        "ac|cabin_pressure",
        "mission.cruise|h0",
    ]
    for name in samples:
        owners = [d.discipline_name for d in disciplines if d.owns(name)]
        assert len(owners) == 1, f"{name} is owned by {owners}"


@pytest.mark.unit
def test_a_discipline_refuses_a_parameter_that_is_not_its_own():
    """A gap in the ownership map is a construction error, not a silent reassignment."""
    aero = Aerodynamics()
    with pytest.raises(DisciplineError, match="not owned by the aerodynamics discipline"):
        aero.add(Parameter("ac|geom|wing|S_ref", 124.6))


@pytest.mark.unit
def test_a_discipline_refuses_the_same_parameter_twice():
    """Declaring a variable twice in a case file means one of the two is being ignored."""
    aero = Aerodynamics()
    aero.add(Parameter("ac|aero|polar|e", 0.801))
    with pytest.raises(DisciplineError, match="already held"):
        aero.add(Parameter("ac|aero|polar|e", 0.85))


@pytest.mark.unit
def test_structures_owns_no_inputs_and_says_why():
    """The honest answer, with a message that does not read as a bug."""
    with pytest.raises(DisciplineError, match="sets nothing"):
        Structures().add(Parameter("ac|geom|wing|S_ref", 124.6))


@pytest.mark.unit
def test_performance_refuses_loose_parameters_and_points_at_the_profile():
    """Mission values belong with the schedules and the ladder that make them reachable."""
    from cdadt import MissionProfile, PhaseSchedule
    from cdadt.mission import STEADY_FLIGHT_PHASES

    profile = MissionProfile(
        parameters={"mission_range": (2800.0, "nmi")},
        schedules={phase: PhaseSchedule(250.0, 0.0) for phase in STEADY_FLIGHT_PHASES},
    )
    performance = Performance(profile, num_nodes=5)
    with pytest.raises(DisciplineError, match="belong in the mission profile"):
        performance.add(Parameter("mission.mission_range", 2800.0))


@pytest.mark.unit
def test_setting_a_parameter_a_discipline_does_not_hold_is_an_error():
    """There is no implicit creation: a typo would make a variable nothing reads."""
    aero = Aerodynamics()
    with pytest.raises(KeyError, match="does not hold"):
        aero.set("ac|aero|polar|e", 0.8)


@pytest.mark.unit
def test_an_aircraft_routes_every_parameter_to_its_owner():
    """The aircraft is a composition, not a flat dictionary."""
    aircraft = Aircraft(
        [
            Parameter("ac|geom|wing|S_ref", 124.6, "m**2"),
            Parameter("ac|aero|polar|e", 0.801),
            Parameter("ac|geom|vstab|AR", 1.91),
            Parameter("ac|weights|W_payload", 18000.0, "kg"),
        ]
    )
    assert aircraft["geometry"].value("ac|geom|wing|S_ref") == 124.6
    assert aircraft["aerodynamics"].value("ac|aero|polar|e") == 0.801
    assert aircraft["stability"].value("ac|geom|vstab|AR") == 1.91
    assert aircraft["weights"].value("ac|weights|W_payload") == 18000.0
    assert aircraft.owner("ac|geom|wing|S_ref").discipline_name == "geometry"
    assert len(aircraft.parameters) == 4


@pytest.mark.unit
def test_an_aircraft_refuses_a_parameter_no_discipline_owns():
    """Either the name is misspelled or the ownership map needs extending; both are errors."""
    with pytest.raises(AircraftError, match="No discipline owns"):
        Aircraft([Parameter("ac|structures|spar_cap_thickness", 0.01)])


@pytest.mark.unit
def test_an_aircraft_reports_which_names_are_unowned():
    """Used by the contract test that walks the whole settable set of a real box."""
    aircraft = Aircraft([])
    assert aircraft.unowned(["ac|geom|wing|AR", "ac|nonsense|thing"]) == ("ac|nonsense|thing",)


@pytest.mark.unit
def test_every_discipline_reports_something():
    """A discipline that neither sets nor reports anything would be an empty box in a report."""
    for discipline in (*AIRCRAFT_DISCIPLINES, Performance):
        assert discipline.reported, f"{discipline.discipline_name} reports nothing"


@pytest.mark.unit
def test_a_discipline_behaves_as_a_collection_of_its_parameters():
    """Membership, iteration and length are how an aircraft walks a discipline."""
    aero = Aerodynamics()
    aero.add(Parameter("ac|aero|polar|e", 0.801, None, "estimate"))
    aero.add(Parameter("ac|aero|Mach_max", 0.82, None, "spec"))

    assert "ac|aero|polar|e" in aero
    assert "ac|geom|wing|AR" not in aero
    assert list(iter(aero)) == ["ac|aero|polar|e", "ac|aero|Mach_max"]
    assert len(aero) == 2
    assert aero.units("ac|aero|Mach_max") is None
    assert aero.parameter("ac|aero|polar|e").source == "estimate"
    assert repr(aero) == "Aerodynamics(name='aerodynamics', parameters=2, responses=2)"


@pytest.mark.unit
def test_collecting_skips_optional_responses_but_raises_on_required_ones():
    """The distinction is what separates "this box does not publish that" from a wiring error."""

    class Fake:
        def __init__(self, published):
            self._published = published

        def get(self, path, units=None):
            return self._published[path]

        def has(self, path):
            return path in self._published

    # Propulsion reports only optional responses, so an empty box yields an empty result.
    propulsion = Propulsion()
    empty = Fake({})
    assert propulsion.collect(empty) == {}
    assert set(propulsion.missing(empty)) == {response.name for response in Propulsion.reported}

    # Aerodynamics reports required ones, so an empty box is an error that names the path.
    with pytest.raises(KeyError, match="which this black box does not publish"):
        Aerodynamics().collect(empty)


@pytest.mark.unit
def test_performance_exposes_the_profile_and_grid_it_owns():
    """A performance discipline detached from its mission could not converge or report it."""
    from cdadt import MissionProfile, PhaseSchedule
    from cdadt.mission import STEADY_FLIGHT_PHASES

    profile = MissionProfile(
        parameters={"mission_range": (2800.0, "nmi")},
        schedules={phase: PhaseSchedule(250.0, 0.0) for phase in STEADY_FLIGHT_PHASES},
    )
    performance = Performance(profile, num_nodes=11)
    assert performance.profile is profile
    assert performance.num_nodes == 11
    assert repr(performance).startswith("Performance(MissionProfile(")


@pytest.mark.unit
def test_applying_performance_writes_the_design_mission():
    """Performance's ``apply`` is the profile's ``apply``, at the grid it was built for."""
    from cdadt import MissionProfile, PhaseSchedule
    from cdadt.mission import STEADY_FLIGHT_PHASES

    written = {}

    class Recording:
        def set(self, name, value, units=None):
            written[name] = (value, units)

        def run(self):
            pass

    profile = MissionProfile(
        parameters={"mission_range": (2800.0, "nmi")},
        schedules={phase: PhaseSchedule(250.0, 0.0) for phase in STEADY_FLIGHT_PHASES},
    )
    Performance(profile, num_nodes=5).apply(Recording())
    assert written["mission.mission_range"] == (2800.0, "nmi")
    assert len(written["mission.cruise.fltcond|Ueas"][0]) == 5
