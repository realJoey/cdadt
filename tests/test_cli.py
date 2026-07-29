"""Integration: the command line does what the API does, and reports it."""

from __future__ import annotations

import json

import pytest

from cdadt.cli import (
    DEFAULT_COMMANDS,
    Command,
    CommandLineInterface,
    InspectCommand,
    OptimizeCommand,
    SizeCommand,
    main,
)
from tests.conftest import CASES


@pytest.mark.unit
def test_the_command_line_refuses_an_unknown_command():
    """argparse exits rather than guessing."""
    with pytest.raises(SystemExit):
        main(["fly", str(CASES / "b738.yaml")])


# =============================================================================================
# The command hierarchy
# =============================================================================================


@pytest.mark.unit
def test_a_command_must_say_what_it_does():
    """``execute`` is abstract, so a half-written command fails at construction, not at run time."""
    with pytest.raises(TypeError, match="abstract"):
        Command()


@pytest.mark.unit
def test_a_new_command_needs_no_change_to_any_existing_one():
    """The open/closed claim, tested rather than asserted in a docstring.

    A command set is composed, not hard-coded, so this adds one without touching ``SizeCommand``,
    ``OptimizeCommand``, ``InspectCommand`` or the dispatch.
    """

    class CountCommand(Command):
        """A command that exists only in this test."""

        name = "count"
        help = "count the design variables"

        def execute(self, arguments) -> int:
            """Print how many design variables the case declares."""
            from cdadt import Config

            print(f"{len(Config.from_yaml(arguments.case).design_variables)} design variables")
            return 0

    interface = CommandLineInterface([*DEFAULT_COMMANDS, CountCommand])

    assert sorted(interface.commands) == ["count", "inspect", "optimize", "size"]
    assert interface.run(["count", str(CASES / "b738.yaml")]) == 0


@pytest.mark.unit
def test_two_commands_with_the_same_name_are_refused():
    """One would silently shadow the other, and which one would depend on ordering."""

    class Duplicate(SizeCommand):
        """Claims a name that is already taken."""

    with pytest.raises(ValueError, match="Two commands are both called 'size'"):
        CommandLineInterface([SizeCommand, Duplicate])


@pytest.mark.unit
def test_the_commands_offered_cannot_be_changed_through_the_accessor():
    """The mapping handed out is a copy, or a caller could remove a command from a live parser."""
    interface = CommandLineInterface()
    interface.commands.clear()

    assert sorted(interface.commands) == ["inspect", "optimize", "size"]


@pytest.mark.unit
def test_the_commands_and_the_interface_repr_as_what_they_are():
    """These appear in tracebacks from a failed study, so they have to name themselves."""
    assert repr(SizeCommand()) == "SizeCommand('size')"
    assert repr(OptimizeCommand()) == "OptimizeCommand('optimize')"
    assert repr(InspectCommand()) == "InspectCommand('inspect')"
    assert repr(CommandLineInterface()) == "CommandLineInterface(['inspect', 'optimize', 'size'])"


@pytest.mark.unit
def test_each_study_command_describes_its_own_verbosity():
    """Sizing prints continuation steps; optimizing also prints driver progress."""
    assert "continuation step" in SizeCommand.verbose_help
    assert "driver progress" in OptimizeCommand.verbose_help


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
    # The case declares both kinds of constraint, and each must be archived as what it is:
    # certification evidence names a regulation and a source, the throttle bands name neither
    # and are plain design constraints. See :mod:`cdadt.certification`.
    declared = {entry["name"]: entry["traceable"] for entry in archived["constraints"]}
    assert declared == {
        "takeoff_field_length": True,
        "engine_out_climb_gradient": True,
        "climb_throttle": False,
        "cruise_throttle": False,
    }
    assert "Design constraints with no stated regulation or source" in printed


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


#: What a sizing run must leave behind, whether or not it was asked to. The first four are
#: cdadt's; ``.openmdao_out`` and ``reports`` are OpenMDAO's, and their presence is the evidence
#: that the run directory really is the problem's own output directory rather than a folder
#: cdadt made beside it.
SIZING_ARTEFACTS = {
    "report.txt",
    "results.json",
    "n2.html",
    "mission.pdf",
    "trajectory.pdf",
    "takeoff.pdf",
    ".openmdao_out",
    "reports",
}


@pytest.mark.integration
def test_a_sizing_run_always_leaves_a_complete_record(tmp_path, capsys):
    """No flag: a study whose files depend on remembering a flag is a study with no record."""
    case = _small_case(tmp_path, "b738.yaml")
    root = tmp_path / "run_outputs"
    assert main(["size", str(case), "--run-outputs", str(root)]) == 0

    runs = list(root.iterdir())
    assert len(runs) == 1, "one invocation, one run directory"
    assert runs[0].name.startswith(f"{case.stem}_"), "the directory is named for the case it ran"
    assert runs[0].name.endswith("_out"), "OpenMDAO's own suffix, because it is OpenMDAO's directory"

    assert {path.name for path in runs[0].iterdir()} == SIZING_ARTEFACTS
    assert all((runs[0] / name).stat().st_size > 0 for name in SIZING_ARTEFACTS - {".openmdao_out", "reports"})

    printed = capsys.readouterr().out
    assert "Run written to" in printed
    # The archived report is the printed one, not a second rendering of it.
    assert (runs[0] / "report.txt").read_text(encoding="utf-8") in printed


@pytest.mark.integration
def test_two_runs_of_one_case_accumulate_rather_than_overwrite(tmp_path):
    """Each run is stamped, so a comparison against last week's answer is still possible."""
    case = _small_case(tmp_path, "b738.yaml")
    root = tmp_path / "run_outputs"

    assert main(["size", str(case), "--run-outputs", str(root)]) == 0
    first = {path.name for path in root.iterdir()}

    # The stamp is to the second, so force a distinct one rather than sleeping for a test.
    next(iter(root.iterdir())).rename(root / "b738_20260101_000000_out")
    assert main(["size", str(case), "--run-outputs", str(root)]) == 0

    assert len(list(root.iterdir())) == 2, f"the second run overwrote the first ({first})"


@pytest.mark.integration
def test_an_optimization_run_leaves_the_drivers_own_log_beside_everything_else(tmp_path, capsys):
    """IPOPT.out lands in the run directory because it *is* the problem's output directory.

    That is the whole reason for using OpenMDAO's mechanism rather than making a folder next to
    it: pyOptSparse writes its log wherever the problem was set up, and nothing cdadt does moves
    it afterwards.
    """

    def shrink(data):
        data["solver"]["maxiter"] = 60
        data["driver"] = {"name": "IPOPT", "maxiter": 5, "tol": 1e-4, "derivative_mode": "fwd"}
        for spec in data["design_variables"].values():
            spec.pop("optimize", None)
        data["design_variables"]["ac|geom|wing|AR"]["optimize"] = {"lower": 9.3, "upper": 9.7}
        data["constraints"] = [{"name": "takeoff_field_length", "upper": 8000.0, "units": "ft"}]

    case = _small_case(tmp_path, "b738_optimization.yaml", shrink)
    root = tmp_path / "run_outputs"
    assert main(["optimize", str(case), "--run-outputs", str(root)]) == 0
    capsys.readouterr()

    run = next(iter(root.iterdir()))
    written = {path.name for path in run.iterdir()}
    assert written >= SIZING_ARTEFACTS, f"missing {SIZING_ARTEFACTS - written}"
    assert "IPOPT.out" in written, "the driver's own log is not in the run directory"

    archived = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert archived["objective"] == "total_fuel"
    assert [entry["name"] for entry in archived["constraints"]] == ["takeoff_field_length"]


@pytest.mark.integration
def test_inspect_can_show_the_outputs_alone(capsys):
    """The other half of the interface, and the half a results reader needs."""
    assert main(["inspect", str(CASES / "b738.yaml"), "--what", "outputs", "--filter", "ac|weights"]) == 0
    printed = capsys.readouterr().out

    assert "Outputs the box publishes" in printed
    assert "Inputs the box accepts" not in printed
    assert "ac|weights|OEW" in printed
