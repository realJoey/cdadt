"""Unit: the aircraft as a composition of disciplines, and the routing that makes it one."""

from __future__ import annotations

import pytest

from cdadt import Aircraft, AircraftError, Geometry, Parameter, Weights
from cdadt.disciplines import AIRCRAFT_DISCIPLINES


class FakeBox:
    """Records what an aircraft writes, and answers what it is asked to read."""

    def __init__(self, published: dict[str, float] | None = None) -> None:
        self.written: dict[str, tuple] = {}
        self._published = published or {}

    def set(self, name, value, units=None):
        """Record a written parameter."""
        self.written[name] = (value, units)

    def get(self, path, units=None):
        """Return a published value."""
        return self._published[path]

    def has(self, path) -> bool:
        """Return whether the fake publishes that path."""
        return path in self._published


def _aircraft() -> Aircraft:
    """Return a small but validly routed aircraft."""
    return Aircraft(
        [
            Parameter("ac|geom|wing|S_ref", 124.6, "m**2"),
            Parameter("ac|geom|wing|AR", 9.45),
            Parameter("ac|weights|W_payload", 18000.0, "kg"),
        ]
    )


# =============================================================================================
# Composition
# =============================================================================================


@pytest.mark.unit
def test_an_aircraft_behaves_as_a_collection_of_its_disciplines():
    """Membership, iteration and length are how a report walks an aircraft."""
    aircraft = _aircraft()
    assert "geometry" in aircraft
    assert "propulsion" in aircraft
    assert "cost" not in aircraft
    assert len(aircraft) == len(AIRCRAFT_DISCIPLINES)
    assert set(iter(aircraft)) == {d.discipline_name for d in AIRCRAFT_DISCIPLINES}
    assert repr(aircraft) == f"Aircraft({len(AIRCRAFT_DISCIPLINES)} disciplines, 3 parameters)"


@pytest.mark.unit
def test_asking_for_a_discipline_that_is_not_there_lists_the_ones_that_are():
    """A typo in a discipline name should not require reading the source to fix."""
    with pytest.raises(KeyError, match="has no 'cost' discipline"):
        _aircraft()["cost"]


@pytest.mark.unit
def test_two_disciplines_cannot_share_a_name():
    """The second would silently replace the first in the composition."""
    with pytest.raises(AircraftError, match="Two disciplines are both called"):
        Aircraft([], disciplines=(Geometry, Geometry))


@pytest.mark.unit
def test_a_variable_claimed_by_two_disciplines_is_refused():
    """Ownership must be disjoint, or the same variable would be written twice.

    Constructed deliberately by composing two disciplines whose patterns overlap, because the
    shipped set does not overlap -- which is itself asserted, against the live model, in
    :mod:`tests.test_boundary`.
    """

    class GreedyWeights(Weights):
        discipline_name = "greedy_weights"
        owned_patterns = ("ac|geom|wing|*",)

    # Routing a parameter names both claimants, which is the actionable message.
    with pytest.raises(AircraftError, match="is claimed by"):
        Aircraft([Parameter("ac|geom|wing|S_ref", 124.6, "m**2")], disciplines=(Geometry, GreedyWeights))

    # Asking who owns it afterwards refuses to pick one of the two.
    aircraft = Aircraft([], disciplines=(Geometry, GreedyWeights))
    with pytest.raises(AircraftError, match="owned by 2 disciplines"):
        aircraft.owner("ac|geom|wing|S_ref")


@pytest.mark.unit
def test_asking_who_owns_something_nobody_owns_is_an_error():
    """Used by the routing itself, so it must fail rather than return None."""
    with pytest.raises(AircraftError, match="owned by 0 disciplines"):
        _aircraft().owner("ac|nonsense|thing")


# =============================================================================================
# State
# =============================================================================================


@pytest.mark.unit
def test_a_value_can_be_read_and_changed_through_the_aircraft():
    """An optimizer changes a design variable; this is that assignment, routed."""
    aircraft = _aircraft()
    assert aircraft.value("ac|geom|wing|AR") == 9.45
    aircraft.set("ac|geom|wing|AR", 11.0)
    assert aircraft.value("ac|geom|wing|AR") == 11.0
    assert aircraft["geometry"].value("ac|geom|wing|AR") == 11.0


# =============================================================================================
# The black-box interface
# =============================================================================================


@pytest.mark.unit
def test_applying_an_aircraft_writes_every_parameter_with_its_units():
    """Units travel with the value; writing a number without them is a unit error waiting."""
    box = FakeBox()
    _aircraft().apply(box)
    assert box.written["ac|geom|wing|S_ref"] == (124.6, "m**2")
    assert box.written["ac|geom|wing|AR"] == (9.45, None)
    assert box.written["ac|weights|W_payload"] == (18000.0, "kg")


@pytest.mark.unit
def test_collecting_returns_one_entry_per_discipline():
    """Grouped by discipline, because that is how a design is read rather than searched."""
    box = FakeBox(
        {
            "ac|weights|MTOW": 78345.6,
            "ac|weights|OEW": 41748.3,
            "ac|weights|MLW": 62676.5,
            "ac|weights|W_payload": 18000.0,
        }
    )
    aircraft = Aircraft([], disciplines=(Weights,))
    collected = aircraft.collect(box)
    assert set(collected) == {"weights"}
    assert collected["weights"]["MTOW"] == 78345.6
    # The optional equipment breakdown is absent from this fake, and is reported as such.
    assert "furnishings_weight" in aircraft.missing(box)["weights"]
