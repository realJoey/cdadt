"""Unit and integration: naming a quantity, and what a run reports."""

from __future__ import annotations

import numpy as np
import pytest

from cdadt import Performance, ResponseCatalog, SizingResults
from cdadt.disciplines import AIRCRAFT_DISCIPLINES


@pytest.mark.unit
def test_the_catalog_maps_short_names_onto_black_box_paths():
    """Which is what lets a case file say ``objective: {name: total_fuel}``."""
    catalog = ResponseCatalog()
    assert catalog.path("total_fuel") == "mission.loiter.fuel_burn_integ.fuel_burn_final"
    assert catalog.path("MTOW") == "ac|weights|MTOW"
    assert catalog.units("takeoff_field_length") == "ft"
    assert catalog.owner("takeoff_field_length") == "performance"


@pytest.mark.unit
def test_an_unknown_name_is_passed_through_as_a_path():
    """A raw black-box path always works; the box gives the error if it is wrong."""
    catalog = ResponseCatalog()
    assert catalog.path("mission.cruise.fltcond|M") == "mission.cruise.fltcond|M"
    assert catalog.units("mission.cruise.fltcond|M") is None


@pytest.mark.unit
def test_the_catalog_covers_every_discipline():
    """A response no catalogue knows cannot be named in a case file."""
    catalog = ResponseCatalog()
    for discipline in (*AIRCRAFT_DISCIPLINES, Performance):
        for response in discipline.reported:
            assert response.name in catalog


@pytest.mark.unit
def test_two_disciplines_cannot_report_different_things_under_one_name():
    """Names are how a case file addresses a response, so they must be unique."""
    from cdadt.disciplines.base import Discipline
    from cdadt.parameters import Response

    class First(Discipline):
        discipline_name = "first"
        reported = (Response("fuel", "a.path", "kg"),)

    class Second(Discipline):
        discipline_name = "second"
        reported = (Response("fuel", "a.different.path", "kg"),)

    with pytest.raises(ValueError, match="different definitions"):
        ResponseCatalog([First, Second])


@pytest.mark.unit
def test_results_can_be_read_flat_or_by_discipline():
    """Both are useful: one for constraints and reports, one for reading a design."""
    results = SizingResults(
        {"weights": {"MTOW": 78345.6}, "performance": {"total_fuel": 18597.3, "climb_throttle": np.arange(3.0)}},
        model="a:Box",
        num_nodes=3,
    )
    assert results["MTOW"] == 78345.6
    assert results.by_discipline["performance"]["total_fuel"] == 18597.3
    assert "climb_throttle" in results
    assert len(results) == 3


@pytest.mark.unit
def test_asking_for_something_that_was_not_reported_lists_what_was():
    """The alternative is a bare KeyError on a name the reader cannot check."""
    results = SizingResults({"weights": {"MTOW": 1.0}})
    with pytest.raises(KeyError, match="Reported:"):
        _ = results["OEW"]


@pytest.mark.unit
def test_only_scalars_are_offered_for_tabulation():
    """Vector histories are results too, but they are not rows in a table."""
    results = SizingResults({"performance": {"total_fuel": 1.0, "climb_throttle": np.arange(5.0)}})
    assert results.scalars() == {"total_fuel": 1.0}


@pytest.mark.unit
def test_results_serialize_without_numpy():
    """A study should be archivable as JSON without re-running it."""
    results = SizingResults(
        {"performance": {"climb_throttle": np.arange(3.0)}},
        missing={"structures": ("wing_weight",)},
        model="a:Box",
        num_nodes=3,
    )
    payload = results.to_dict()
    assert payload["results"]["performance"]["climb_throttle"] == [0.0, 1.0, 2.0]
    assert payload["unavailable"] == {"structures": ["wing_weight"]}


@pytest.mark.integration
def test_a_real_run_reports_every_discipline(converged_analysis):
    """Reading the whole set is what makes a run report complete."""
    results = converged_analysis.results()
    assert set(results.by_discipline) == {
        "geometry",
        "aerodynamics",
        "propulsion",
        "stability",
        "structures",
        "weights",
        "performance",
    }
    assert not results.missing, results.missing


@pytest.mark.integration
def test_the_report_names_the_black_box_that_produced_it(converged_analysis):
    """A table of numbers with no model attached is not a result."""
    report = converged_analysis.results().report(converged_analysis.catalog)
    assert "openconcept.examples.B738_sizing:B738SizingMissionAnalysis" in report
    assert "21 nodes per phase" in report
    assert "structures" in report and "performance" in report


@pytest.mark.integration
def test_the_structural_breakdown_is_a_breakdown_of_the_structural_total(converged_analysis):
    """Every reported component is positive, and together they account for the total.

    The box adds an allowance on top of the component sum, so the total is larger than the sum
    rather than equal to it. The size of that allowance is a constant inside the box and is
    deliberately not asserted here: pinning it would be cdadt asserting somebody else's
    internal, which is exactly what the black-box boundary exists to avoid.
    """
    results = converged_analysis.results()
    components = [
        "wing_weight",
        "hstab_weight",
        "vstab_weight",
        "fuselage_weight",
        "main_gear_weight",
        "nose_gear_weight",
        "nacelle_weight",
    ]
    for name in components:
        assert results[name] > 0.0, name
    total = sum(results[name] for name in components)
    assert total < results["structure_weight"] < 2.0 * total
    # And structure is the dominant part of the empty weight it is reported alongside.
    assert 0.3 < results["structure_weight"] / results["OEW"] < 0.9


@pytest.mark.unit
def test_the_catalog_names_what_is_available_when_asked_for_something_that_is_not():
    """A bare KeyError on a response name would not tell a reader what to write instead."""
    catalog = ResponseCatalog()
    with pytest.raises(KeyError, match="Available:"):
        catalog.response("specific_air_range")


@pytest.mark.unit
def test_the_catalog_and_results_repr_as_their_size():
    """Both appear in debugger frames while a run is being inspected."""
    assert repr(ResponseCatalog()).startswith("ResponseCatalog(")
    results = SizingResults({"weights": {"MTOW": 1.0}}, model="a:Box", num_nodes=3)
    assert repr(results) == "SizingResults(1 disciplines, 1 responses)"


@pytest.mark.unit
def test_results_expose_the_run_they_came_from():
    """A results object detached from its model and grid cannot be compared to another."""
    results = SizingResults({"weights": {"MTOW": 1.0}}, model="a:Box", num_nodes=21)
    assert results.model == "a:Box"
    assert results.num_nodes == 21
    assert list(iter(results)) == ["MTOW"]


@pytest.mark.unit
def test_the_report_skips_empty_disciplines_and_names_what_was_unavailable():
    """A discipline with nothing to say prints nothing; an unavailable response is stated.

    Silently omitting a response the box did not publish would read as a discipline that found
    nothing to report, which is a different claim entirely.
    """
    results = SizingResults(
        {"weights": {"MTOW": 78345.6}, "structures": {}},
        missing={"structures": ("wing_weight", "fuselage_weight")},
        model="a:Box",
        num_nodes=3,
    )
    report = results.report()
    assert "MTOW" in report
    assert "Not published by this black box" in report
    assert "structures: wing_weight, fuselage_weight" in report
