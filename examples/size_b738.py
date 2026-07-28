"""Size a Boeing 737-800 against its design mission and Part 25 reserves.

Runs the analysis only -- no optimizer, no constraints -- and prints the results. This is the
case cdadt is validated on: with the default phase model it reproduces OpenConcept's own
``B738_sizing.py`` example to within solver tolerance.

Usage::

    python examples/size_b738.py
    python examples/size_b738.py --num-nodes 11      # faster, slightly coarser integration
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cdadt import AircraftDefinition, MissionProfile, SizingAnalysis

CASES = Path(__file__).resolve().parent.parent / "cases"


def main() -> None:
    """Parse arguments, run the sizing analysis, and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aircraft", type=Path, default=CASES / "b738_aircraft.yaml")
    parser.add_argument("--mission", type=Path, default=CASES / "b738_mission.yaml")
    parser.add_argument("--num-nodes", type=int, default=21, help="Analysis points per phase; must be odd")
    parser.add_argument("--quiet", action="store_true", help="Do not print the continuation steps")
    args = parser.parse_args()

    analysis = SizingAnalysis(
        aircraft=AircraftDefinition.from_yaml(args.aircraft),
        profile=MissionProfile.from_yaml(args.mission),
        num_nodes=args.num_nodes,
    )
    analysis.run(verbose=not args.quiet)
    print()
    print(analysis.report())


if __name__ == "__main__":
    main()
