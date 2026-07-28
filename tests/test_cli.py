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
    assert "Constraints" in printed
    assert "14 CFR 25.121(b)(1)(i)" in printed
    assert "SUCCEEDED" in printed

    archived = json.loads(destination.read_text(encoding="utf-8"))
    assert archived["succeeded"] is True
    assert archived["objective"] == "total_fuel"
    assert {entry["status"] for entry in archived["constraints"]} <= {"MET", "ACTIVE"}
    assert all(entry["traceable"] for entry in archived["constraints"])


@pytest.mark.integration
def test_the_module_entry_point_runs_the_same_command_line(monkeypatch, capsys):
    """``python -m cdadt`` must be the same program as the ``cdadt`` console script.

    Executed through :func:`runpy.run_module` rather than a subprocess, so the entry point is
    genuinely exercised in this interpreter rather than merely observed to exit zero.
    """
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["cdadt", "--version"])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_module("cdadt", run_name="__main__")

    assert exit_status.value.code == 0
    assert "cdadt" in capsys.readouterr().out


def _small_case(tmp_path, name: str, mutate=None):
    """Write a shipped case to ``tmp_path`` on a coarse grid, optionally mutated."""
    import yaml

    from tests.conftest import case_dict

    data = case_dict(name)
    data["black_box"]["num_nodes"] = 5
    if mutate is not None:
        mutate(data)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.integration
def test_size_prints_without_being_asked_to_archive(tmp_path, capsys):
    """``--json`` is optional; the report is the primary output and stands alone."""
    case = _small_case(tmp_path, "b738.yaml")
    assert main(["size", str(case)]) == 0

    printed = capsys.readouterr().out
    assert "performance" in printed
    assert "takeoff_field_length" in printed
    assert "Results written to" not in printed
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.integration
def test_optimize_prints_without_being_asked_to_archive(tmp_path, capsys):
    """Same for an optimization: the traceability matrix is the output that matters."""

    def shrink(data):
        data["solver"]["maxiter"] = 60
        data["driver"] = {"name": "IPOPT", "maxiter": 5, "tol": 1e-4, "derivative_mode": "fwd"}
        for spec in data["design_variables"].values():
            spec.pop("optimize", None)
        data["design_variables"]["ac|geom|wing|AR"]["optimize"] = {"lower": 9.3, "upper": 9.7}
        data["constraints"] = [
            {
                "name": "takeoff_field_length",
                "upper": 8000.0,
                "units": "ft",
                "regulation": "14 CFR 25.113",
                "source": "8000 ft dry runway at sea level, ISA",
            }
        ]

    case = _small_case(tmp_path, "b738_optimization.yaml", shrink)
    assert main(["optimize", str(case)]) == 0

    printed = capsys.readouterr().out
    assert "Constraints" in printed
    assert "Results written to" not in printed
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.integration
def test_inspect_can_show_the_outputs_alone(capsys):
    """The other half of the interface, and the half a results reader needs."""
    assert main(["inspect", str(CASES / "b738.yaml"), "--what", "outputs", "--filter", "ac|weights"]) == 0
    printed = capsys.readouterr().out

    assert "Outputs the box publishes" in printed
    assert "Inputs the box accepts" not in printed
    assert "ac|weights|OEW" in printed
