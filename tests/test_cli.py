"""Integration: the command line does what the API does, and reports it."""

from __future__ import annotations

import json

import pytest

from cdadt.cli import main
from tests.conftest import CASES


@pytest.mark.unit
def test_the_command_line_refuses_an_unknown_command():
    """argparse exits rather than guessing."""
    with pytest.raises(SystemExit):
        main(["fly", str(CASES / "b738.yaml")])


@pytest.mark.integration
def test_inspect_prints_the_interface_without_running_anything(capsys):
    """The interface reference, generated rather than transcribed."""
    assert main(["inspect", str(CASES / "b738.yaml")]) == 0
    printed = capsys.readouterr().out

    assert "openconcept.examples.B738_sizing:B738SizingMissionAnalysis" in printed
    assert "Inputs the box accepts" in printed
    assert "Outputs the box publishes" in printed
    # An input, and an output that is emphatically not an input.
    assert "ac|geom|wing|S_ref" in printed
    assert "ac|weights|OEW" in printed


@pytest.mark.integration
def test_inspect_can_be_filtered(capsys):
    """A 700-line interface listing is not a reference anybody reads."""
    assert main(["inspect", str(CASES / "b738.yaml"), "--what", "inputs", "--filter", "ac|geom|wing"]) == 0
    printed = capsys.readouterr().out

    assert "ac|geom|wing|S_ref" in printed
    assert "ac|aero|polar|e" not in printed
    assert "Outputs the box publishes" not in printed


@pytest.mark.integration
@pytest.mark.slow
def test_size_runs_the_case_and_can_archive_it(tmp_path, capsys):
    """The report is printed, and the same numbers are written to JSON."""
    destination = tmp_path / "b738.json"
    assert main(["size", str(CASES / "b738.yaml"), "--json", str(destination)]) == 0
    printed = capsys.readouterr().out

    assert "performance" in printed
    assert "takeoff_field_length" in printed

    archived = json.loads(destination.read_text(encoding="utf-8"))
    assert archived["model"] == "openconcept.examples.B738_sizing:B738SizingMissionAnalysis"
    assert archived["num_nodes"] == 21
    assert archived["results"]["weights"]["MTOW"] == pytest.approx(78345.6435, rel=1e-6)
    assert archived["results"]["performance"]["takeoff_field_length"] == pytest.approx(5247.7948, rel=1e-6)
    # Vector results survive the round trip as lists, not as unserializable arrays.
    assert len(archived["results"]["performance"]["climb_throttle"]) == 21


@pytest.mark.integration
@pytest.mark.slow
def test_optimize_runs_the_study_and_reports_the_traceability_matrix(tmp_path, capsys):
    """And exits non-zero if it did not succeed, so a shell cannot archive a failure as one."""
    destination = tmp_path / "b738_optimization.json"
    status = main(["optimize", str(CASES / "b738_optimization.yaml"), "--json", str(destination)])
    printed = capsys.readouterr().out

    assert status == 0, printed
    assert "Certification basis" in printed
    assert "14 CFR 25.121(b)(1)(i)" in printed
    assert "SUCCEEDED" in printed

    archived = json.loads(destination.read_text(encoding="utf-8"))
    assert archived["succeeded"] is True
    assert archived["objective"] == "total_fuel"
    assert {entry["status"] for entry in archived["requirements"]} <= {"MET", "ACTIVE"}
    assert all(entry["regulation"] and entry["source"] for entry in archived["requirements"])
