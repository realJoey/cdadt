"""Size a Boeing 737-800 against a 2800 nmi mission with Part 25 reserves.

The whole study is in ``cases/b738.yaml``; this script exists to show the three lines of API
underneath the command line. ``python -m cdadt size cases/b738.yaml`` does the same thing.

Run::

    python examples/size_b738.py
"""

from pathlib import Path

from cdadt import Config, SizingAnalysis

CASE = Path(__file__).resolve().parent.parent / "cases" / "b738.yaml"


def main() -> None:
    """Converge the shipped B738 case and print every response."""
    analysis = SizingAnalysis(Config.from_yaml(CASE))
    results = analysis.run(verbose=True)
    print(results.report(analysis.catalog))


if __name__ == "__main__":
    main()
