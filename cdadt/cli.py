"""The command line: size a case, optimize a case, or read the interface of its black box.

Three commands, all taking a case file:

``cdadt size <case>``
    Converge the aircraft against its mission and print every response.

``cdadt optimize <case>``
    Converge a baseline, drive the design variables against the certification basis, and print
    the comparison and the traceability matrix.

``cdadt inspect <case>``
    Print what the case's black box accepts and what it publishes, without running anything.
    This is the interface reference, generated rather than transcribed -- which is the only kind
    that stays true.

Each command can write its results to JSON with ``--json``, so a study is archivable without
re-running it, and ``--outputs <dir>`` additionally leaves the model diagram, the flown
trajectory and the printed report in one directory. See :mod:`cdadt.artifacts`.

**The shape of this module.** Every command is a :class:`Command` subclass that declares its own
name, its own arguments and what it does, and :class:`CommandLineInterface` composes whichever
set it is given. A fourth command is therefore a fourth subclass and one entry in
:data:`DEFAULT_COMMANDS`; no existing command, and no dispatch table, is touched. That is the
same open/closed arrangement the disciplines use, and it is why there is no ``if command ==``
anywhere here.

Nothing in this module computes anything. A command reads a case file, hands it to the classes
that do the work, and turns what comes back into text and an exit status.
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from cdadt import __version__
from cdadt.analysis import SizingAnalysis
from cdadt.artifacts import StudyArtifacts
from cdadt.blackbox import OpenConceptSizingBox, RunDirectory
from cdadt.config import Config
from cdadt.optimization import Optimizer

__all__ = [
    "DEFAULT_COMMANDS",
    "Command",
    "CommandLineInterface",
    "InspectCommand",
    "OptimizeCommand",
    "SizeCommand",
    "StudyCommand",
    "main",
]


class Command(ABC):
    """One subcommand of ``cdadt``.

    A subclass declares what it is called and what it does; the interface that composes it owns
    the parser and the dispatch, so a command never needs to know it has siblings.

    Attributes
    ----------
    name : str
        The word typed after ``cdadt``.
    help : str
        One line, shown in ``cdadt --help``.
    """

    name: ClassVar[str]
    help: ClassVar[str]

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Declare this command's arguments on its own subparser.

        The base declares the one argument every command takes -- the case file -- so a subclass
        that needs nothing else does not override this at all.
        """
        parser.add_argument("case", type=Path, help="path to the case file")

    @abstractmethod
    def execute(self, arguments: argparse.Namespace) -> int:
        """Run the command and return the process exit status."""

    def __repr__(self) -> str:
        """Return a representation naming the command."""
        return f"{type(self).__name__}({self.name!r})"


class StudyCommand(Command):
    """A command that runs a study and leaves a complete record of it behind.

    Both studies -- sizing and optimizing -- write the same files into the same place, so that
    behaviour lives here once. What differs is only what is run and what the archived payload
    contains, which is what the subclasses supply.

    **Every run writes its outputs.** Not on request: a study whose files depend on a flag is a
    study whose record depends on remembering the flag. The run directory is
    :class:`~cdadt.blackbox.RunDirectory`, which is OpenMDAO's own mechanism, so the driver's log
    and OpenMDAO's reports land beside cdadt's files rather than in the working directory.
    """

    verbose_help: ClassVar[str] = "print each continuation step"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Declare the case file, where runs go, and the verbosity flag."""
        super().configure(parser)
        parser.add_argument(
            "--run-outputs",
            type=Path,
            default=Path(RunDirectory.DEFAULT_ROOT),
            help=f"where run directories are created (default: {RunDirectory.DEFAULT_ROOT})",
        )
        parser.add_argument("--json", type=Path, default=None, help="also write the results to this JSON file")
        parser.add_argument("-v", "--verbose", action="store_true", help=self.verbose_help)

    @staticmethod
    def run_directory(arguments: argparse.Namespace) -> RunDirectory:
        """Return the run directory this invocation writes into.

        Named for the case and the moment it was run, so repeated runs of one case accumulate
        instead of overwriting each other. The clock is read here, at the edge, and the stamp is
        passed in as data -- everything below this point is reproducible given the same stamp.
        """
        return RunDirectory.for_case(
            arguments.case,
            stamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            root=arguments.run_outputs,
        )

    def archive(
        self,
        arguments: argparse.Namespace,
        analysis: SizingAnalysis,
        report: str,
        payload: dict[str, Any],
    ) -> None:
        """Write the report, the numbers and the three figures into the run directory."""
        if arguments.json:
            arguments.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Results written to {arguments.json}")

        artifacts = StudyArtifacts(analysis.box.run_directory.path)
        title = f"{arguments.case.stem} -- {analysis.box.model_spec}"
        mission_path = analysis.config.mission_path

        artifacts.write_text(report, "report.txt")
        artifacts.write_json(payload, "results.json")
        artifacts.write_n2(analysis.box)
        artifacts.write_mission_profile(analysis.box, title=title, mission_path=mission_path)
        artifacts.write_trajectory(analysis.box, title=title, mission_path=mission_path)
        artifacts.write_takeoff(analysis.box, title=title, mission_path=mission_path)

        print(f"\nRun written to {artifacts.directory}")
        for path in artifacts.written():
            print(f"  {path.name}")


class SizeCommand(StudyCommand):
    """Converge a case and print every response."""

    name: ClassVar[str] = "size"
    help: ClassVar[str] = "converge a case and report every response"

    def execute(self, arguments: argparse.Namespace) -> int:
        """Run a sizing case and print the results."""
        analysis = SizingAnalysis(Config.from_yaml(arguments.case), run=self.run_directory(arguments))
        results = analysis.run(verbose=arguments.verbose)
        report = results.report(analysis.catalog)
        print(report)
        self.archive(arguments, analysis, report, results.to_dict())
        return 0


class OptimizeCommand(StudyCommand):
    """Optimize a case against its certification basis."""

    name: ClassVar[str] = "optimize"
    help: ClassVar[str] = "optimize a case against its certification basis"
    verbose_help: ClassVar[str] = "print continuation and driver progress"

    def execute(self, arguments: argparse.Namespace) -> int:
        """Run an optimization case and print the comparison and traceability matrix."""
        analysis = SizingAnalysis(Config.from_yaml(arguments.case), run=self.run_directory(arguments))
        optimizer = Optimizer(analysis)
        outcome = optimizer.run(verbose=arguments.verbose)
        report = optimizer.report(outcome)
        print(report)
        self.archive(arguments, analysis, report, self._payload(outcome))
        # A study that did not converge, or that ends on a violated constraint, is not a success,
        # and a shell that treats it as one will archive it as one.
        return 0 if outcome.succeeded else 1

    @staticmethod
    def _payload(outcome: Any) -> dict[str, Any]:
        """Return the full record of an optimization: both designs and every constraint."""
        return {
            "objective": outcome.objective,
            "sense": outcome.sense,
            "succeeded": outcome.succeeded,
            "baseline": outcome.baseline.to_dict(),
            "optimum": outcome.optimum.to_dict(),
            "constraints": [
                {
                    "name": result.constraint.name,
                    "title": result.constraint.title,
                    "regulation": result.constraint.regulation,
                    "source": result.constraint.source,
                    "traceable": result.constraint.is_traceable,
                    "bound": result.constraint.spec.bounds.describe(),
                    "units": result.constraint.units,
                    "value": result.value,
                    "margin": result.margin,
                    "status": result.status,
                }
                for result in outcome.constraints
            ],
        }


class InspectCommand(Command):
    """Print the interface of a case's black box without running it."""

    name: ClassVar[str] = "inspect"
    help: ClassVar[str] = "print the interface of the case's black box"

    HALVES: ClassVar[tuple[str, ...]] = ("inputs", "outputs", "both")

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Declare the case file and which half of the interface to print."""
        super().configure(parser)
        parser.add_argument(
            "--what",
            choices=self.HALVES,
            default="both",
            help="which half of the interface to print (default: both)",
        )
        parser.add_argument("--filter", default="", help="only show names containing this text")

    def execute(self, arguments: argparse.Namespace) -> int:
        """Print what the case's black box accepts and publishes."""
        config = Config.from_yaml(arguments.case)
        box = OpenConceptSizingBox.describe(config.black_box.model, options=config.black_box.options)

        print(f"Black box : {config.black_box.model}")
        print(f"Case      : {arguments.case}")
        print()

        if arguments.what in ("inputs", "both"):
            self._print_half("Inputs the box accepts", box.settable(), arguments.filter)
            print()

        if arguments.what in ("outputs", "both"):
            self._print_half("Outputs the box publishes", box.readable(), arguments.filter)

        return 0

    @staticmethod
    def _print_half(title: str, variables: dict[str, Any], text: str) -> None:
        """Print one half of the interface, saying how much of it the filter kept."""
        shown = {name: info for name, info in variables.items() if text in name}
        print(f"{title} ({len(shown)} of {len(variables)} shown)")
        print("-" * 72)
        for name, info in shown.items():
            print(f"  {name:<52s} {info.units or '-'!s:<10s} {info.shape}")


#: The commands ``cdadt`` offers. A new one is a new class and one entry here.
DEFAULT_COMMANDS: tuple[type[Command], ...] = (SizeCommand, OptimizeCommand, InspectCommand)


class CommandLineInterface:
    """The ``cdadt`` command line: a parser, a set of commands, and the dispatch between them.

    Parameters
    ----------
    commands : sequence of type, optional
        Command classes to offer. Default :data:`DEFAULT_COMMANDS`. Passing a different set is
        how a study adds a command, and is what lets the dispatch be tested without running one.

    Raises
    ------
    ValueError
        If two commands claim the same name, which would otherwise mean one silently shadowed
        the other.

    Examples
    --------
    >>> CommandLineInterface().run(["--version"])
    Traceback (most recent call last):
    SystemExit: 0
    """

    def __init__(self, commands: Sequence[type[Command]] = DEFAULT_COMMANDS) -> None:
        self._commands: dict[str, Command] = {}
        for command_class in commands:
            command = command_class()
            if command.name in self._commands:
                raise ValueError(f"Two commands are both called '{command.name}'.")
            self._commands[command.name] = command

    @property
    def commands(self) -> dict[str, Command]:
        """The commands offered, by name."""
        return dict(self._commands)

    def parser(self) -> argparse.ArgumentParser:
        """Return the parser for the whole command line, with every command configured on it."""
        parser = argparse.ArgumentParser(
            prog="cdadt",
            description="A certification-driven aircraft design tool. Drives an OpenConcept "
            "full-mission sizing analysis as a black box.",
        )
        parser.add_argument("--version", action="version", version=f"cdadt {__version__}")
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in self._commands.values():
            command.configure(subparsers.add_parser(command.name, help=command.help))
        return parser

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Parse ``argv`` and run the command it names, returning the exit status."""
        arguments = self.parser().parse_args(argv)
        return self._commands[arguments.command].execute(arguments)

    def __repr__(self) -> str:
        """Return a representation naming the commands offered."""
        return f"CommandLineInterface({sorted(self._commands)})"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    The console-script entry point, and nothing more: the work is
    :class:`CommandLineInterface`, which is what a test or another tool should use.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status. Non-zero when an optimization did not succeed, or when the case
        could not be run at all.
    """
    return CommandLineInterface().run(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
