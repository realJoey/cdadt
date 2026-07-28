"""Size and optimize a Boeing 737-800 against its certification basis.

Run with::

    python examples/optimize_b738.py

Minimizes total mission fuel -- design mission plus Part 25 reserves -- subject to balanced
field length, one-engine-inoperative climb gradient, landing field length, approach speed
and throttle margin, with wing planform and engine rating free.

Everything about the case lives in ``configs/b738.yml``: the airframe, the mission, the
certification limits and their sources, the design variable bounds, and the optimizer
settings. This script chooses only what cannot be a number -- which providers, which engine
deck, which requirements are active, and what to minimize.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cdadt.aircraft import jet_transport_disciplines
from cdadt.certification import (
    ApproachSpeedLimit,
    BalancedFieldLength,
    CertificationBasis,
    EngineOutClimbGradient,
    LandingFieldLengthLimit,
    ThrottleMargin,
)
from cdadt.core import AircraftConfiguration
from cdadt.optimization import DesignProblem, IpoptDriver, SlsqpDriver

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "b738.yml"

#: Phases whose throttle is constrained. The takeoff phases run at fixed throttle by
#: construction, so constraining them would constrain a constant.
THROTTLED_PHASES = ("climb", "cruise", "descent", "reserve_climb", "reserve_cruise", "loiter")


def build_problem(config_path, optimizer, objective, num_nodes):
    """Assemble the B738 design problem.

    Parameters
    ----------
    config_path : pathlib.Path
        Configuration file.
    optimizer : str
        ``"IPOPT"`` or ``"SLSQP"``.
    objective : str
        Quantity to minimize.
    num_nodes : int
        Analysis points per mission phase.

    Returns
    -------
    DesignProblem
        The assembled problem.
    """
    config = AircraftConfiguration.from_yaml(config_path)

    certification = CertificationBasis(
        [
            BalancedFieldLength(config),
            EngineOutClimbGradient(config),
            LandingFieldLengthLimit(config),
            ApproachSpeedLimit(config),
            *[ThrottleMargin(config, phase=phase) for phase in THROTTLED_PHASES],
        ]
    )

    driver_class = {"IPOPT": IpoptDriver, "SLSQP": SlsqpDriver}[optimizer]

    return DesignProblem(
        disciplines=jet_transport_disciplines(config, engine_deck="CFM56"),
        config=config,
        certification=certification,
        driver=driver_class(config),
        objective=objective,
        num_nodes=num_nodes,
    )


def main():
    """Parse arguments, run the optimization, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="configuration file")
    parser.add_argument("--optimizer", choices=["IPOPT", "SLSQP"], default="IPOPT")
    parser.add_argument("--objective", default="total_fuel", help="quantity to minimize")
    parser.add_argument("--num-nodes", type=int, default=11, help="analysis points per phase (odd)")
    parser.add_argument("--verbose", action="store_true", help="print the continuation schedule")
    args = parser.parse_args()

    design_problem = build_problem(args.config, args.optimizer, args.objective, args.num_nodes)
    print(f"{design_problem}\n")

    problem = design_problem.build()
    result = design_problem.run(problem, verbose=args.verbose)
    print(result.report())

    return 0 if result.optimal else 1


if __name__ == "__main__":
    raise SystemExit(main())
