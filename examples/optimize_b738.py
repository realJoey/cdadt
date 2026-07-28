"""Optimize a Boeing 737-800 against an explicit Part 25 certification basis.

Minimizes fuel with reserves over the wing planform and the engine rating, subject to the
balanced field length, the engine-out second-segment climb gradient and the throttle limits of
the engine deck. Every one of those -- the design variables, the objective and the requirements
-- is declared in ``cases/b738_optimization.yaml``, not here.

``python -m cdadt optimize cases/b738_optimization.yaml`` does the same thing.

Run::

    python examples/optimize_b738.py
"""

from pathlib import Path

from cdadt import Config, Optimizer, SizingAnalysis

CASE = Path(__file__).resolve().parent.parent / "cases" / "b738_optimization.yaml"


def main() -> None:
    """Optimize the shipped B738 case and print the comparison and traceability matrix."""
    optimizer = Optimizer(SizingAnalysis(Config.from_yaml(CASE)))
    outcome = optimizer.run()
    print(optimizer.report(outcome))


if __name__ == "__main__":
    main()
