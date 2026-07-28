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
re-running it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cdadt import __version__
from cdadt.analysis import SizingAnalysis
from cdadt.blackbox import OpenConceptSizingBox
from cdadt.config import Config
from cdadt.optimization import Optimizer

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the whole command line."""
    parser = argparse.ArgumentParser(
        prog="cdadt",
        description="A certification-driven aircraft design tool. Drives an OpenConcept "
        "full-mission sizing analysis as a black box.",
    )
    parser.add_argument("--version", action="version", version=f"cdadt {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    size = commands.add_parser("size", help="converge a case and report every response")
    size.add_argument("case", type=Path, help="path to the case file")
    size.add_argument("--json", type=Path, default=None, help="also write the results to this JSON file")
    size.add_argument("-v", "--verbose", action="store_true", help="print each continuation step")

    optimize = commands.add_parser("optimize", help="optimize a case against its certification basis")
    optimize.add_argument("case", type=Path, help="path to the case file")
    optimize.add_argument("--json", type=Path, default=None, help="also write the results to this JSON file")
    optimize.add_argument("-v", "--verbose", action="store_true", help="print continuation and driver progress")

    inspect = commands.add_parser("inspect", help="print the interface of the case's black box")
    inspect.add_argument("case", type=Path, help="path to the case file")
    inspect.add_argument(
        "--what",
        choices=("inputs", "outputs", "both"),
        default="both",
        help="which half of the interface to print (default: both)",
    )
    inspect.add_argument("--filter", default="", help="only show names containing this text")

    return parser


def _size(arguments: argparse.Namespace) -> int:
    """Run a sizing case and print the results."""
    analysis = SizingAnalysis(Config.from_yaml(arguments.case))
    results = analysis.run(verbose=arguments.verbose)
    print(results.report(analysis.catalog))
    if arguments.json:
        arguments.json.write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
        print(f"Results written to {arguments.json}")
    return 0


def _optimize(arguments: argparse.Namespace) -> int:
    """Run an optimization case and print the comparison and traceability matrix."""
    analysis = SizingAnalysis(Config.from_yaml(arguments.case))
    optimizer = Optimizer(analysis)
    outcome = optimizer.run(verbose=arguments.verbose)
    print(optimizer.report(outcome))
    if arguments.json:
        payload = {
            "objective": outcome.objective,
            "sense": outcome.sense,
            "succeeded": outcome.succeeded,
            "baseline": outcome.baseline.to_dict(),
            "optimum": outcome.optimum.to_dict(),
            "requirements": [
                {
                    "name": result.requirement.name,
                    "regulation": result.requirement.regulation,
                    "title": result.requirement.title,
                    "source": result.requirement.source,
                    "limit": result.requirement.limit,
                    "sense": result.requirement.sense,
                    "value": result.value,
                    "margin": result.margin,
                    "status": result.status,
                }
                for result in outcome.requirements
            ],
        }
        arguments.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {arguments.json}")
    # A study that did not converge, or that ends on a violated requirement, is not a success,
    # and a shell that treats it as one will archive it as one.
    return 0 if outcome.succeeded else 1


def _inspect(arguments: argparse.Namespace) -> int:
    """Print what the case's black box accepts and publishes."""
    config = Config.from_yaml(arguments.case)
    box = OpenConceptSizingBox.describe(config.black_box.model)

    print(f"Black box : {config.black_box.model}")
    print(f"Case      : {arguments.case}")
    print()

    if arguments.what in ("inputs", "both"):
        settable = {n: v for n, v in box.settable().items() if arguments.filter in n}
        print(f"Inputs the box accepts ({len(settable)} of {len(box.settable())} shown)")
        print("-" * 72)
        for name, info in settable.items():
            print(f"  {name:<52s} {info.units or '-'!s:<10s} {info.shape}")
        print()

    if arguments.what in ("outputs", "both"):
        readable = {n: v for n, v in box.readable().items() if arguments.filter in n}
        print(f"Outputs the box publishes ({len(readable)} of {len(box.readable())} shown)")
        print("-" * 72)
        for name, info in readable.items():
            print(f"  {name:<52s} {info.units or '-'!s:<10s} {info.shape}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

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
    arguments = _parser().parse_args(argv)
    commands = {"size": _size, "optimize": _optimize, "inspect": _inspect}
    return commands[arguments.command](arguments)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
