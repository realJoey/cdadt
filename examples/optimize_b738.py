"""Optimize a Boeing 737-800 against an explicit Part 25 certification basis.

Minimizes total mission fuel -- design mission plus reserves -- over the wing planform and the
engine rating, subject to four requirements: the balanced field length must fit the runway, the
engine-out second-segment climb gradient must meet 14 CFR 25.121(b), the throttle must stay
within its limit, and the reference approach speed must stay inside the aerodrome category.

The phase model is :class:`~cdadt.model.Part25PhaseModel`, which deploys takeoff flaps for the
engine-out climb condition as §25.121(b) specifies. That is *not* what OpenConcept's own
example does, and it lowers the achieved gradient; the difference is deliberate and is
discussed in ``docs/validation.rst``.

Usage::

    python examples/optimize_b738.py
    python examples/optimize_b738.py --objective MTOW --optimizer SLSQP
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cdadt import (
    AircraftDefinition,
    ApproachSpeed,
    BalancedFieldLength,
    CertificationBasis,
    DesignOptimizer,
    DesignVariable,
    EngineOutClimbGradient,
    MissionProfile,
    Part25PhaseModel,
    SizingAnalysis,
    ThrottleMargin,
)

CASES = Path(__file__).resolve().parent.parent / "cases"

#: What the optimizer may change. Bounds are the range over which the empirical weight and
#: drag correlations underneath are defensible for a transport of this class, not the range
#: over which the code runs.
DESIGN_VARIABLES = (
    DesignVariable("ac|geom|wing|S_ref", lower=90.0, upper=180.0, units="m**2"),
    DesignVariable("ac|geom|wing|AR", lower=7.0, upper=13.0),
    DesignVariable("ac|geom|wing|c4sweep", lower=15.0, upper=32.0, units="deg"),
    DesignVariable("ac|geom|wing|taper", lower=0.12, upper=0.35),
    DesignVariable("ac|propulsion|engine|rating", lower=18000.0, upper=34000.0, units="lbf"),
)


def certification_basis() -> CertificationBasis:
    """Return the requirements this design is certified against.

    Every limit is stated with its source. None of them is a default: which runway, which
    aerodrome category and which engine count all belong to the operating case, not to the
    code.
    """
    return CertificationBasis(
        [
            BalancedFieldLength(
                limit=8000.0,
                source="Design field length, 8000 ft dry runway at sea level, ISA",
            ),
            EngineOutClimbGradient(
                limit=0.024,
                source="14 CFR 25.121(b)(1)(i), two-engine aeroplane",
            ),
            ApproachSpeed(
                limit=140.0,
                source="ICAO Doc 8168 / 14 CFR 97 approach category C upper bound, 140 kn",
            ),
            ThrottleMargin(
                phase="climb",
                limit=1.0,
                source="Engine deck rated condition; the CFM56 surrogate is not fitted above throttle 1",
            ),
            ThrottleMargin(
                phase="cruise",
                limit=1.0,
                source="Engine deck rated condition; the CFM56 surrogate is not fitted above throttle 1",
            ),
        ]
    )


def main() -> None:
    """Parse arguments, run the optimization, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aircraft", type=Path, default=CASES / "b738_aircraft.yaml")
    parser.add_argument("--mission", type=Path, default=CASES / "b738_mission.yaml")
    parser.add_argument("--num-nodes", type=int, default=11, help="Analysis points per phase; must be odd")
    parser.add_argument("--objective", default="total_fuel", help="A SizingAnalysis result key")
    parser.add_argument("--optimizer", default="IPOPT", help="IPOPT, SNOPT (pyOptSparse) or SLSQP (SciPy)")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    analysis = SizingAnalysis(
        aircraft=AircraftDefinition.from_yaml(args.aircraft),
        profile=MissionProfile.from_yaml(args.mission),
        phase_model=Part25PhaseModel,
        num_nodes=args.num_nodes,
    )

    optimizer = DesignOptimizer(
        analysis=analysis,
        design_variables=DESIGN_VARIABLES,
        objective=args.objective,
        certification=certification_basis(),
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        # SLSQP tests convergence on the scaled objective, so a mass in kilograms needs a
        # reference value. IPOPT scales from its own gradient and ignores this.
        objective_ref=2.0e4,
    )

    optimizer.run(verbose=not args.quiet)
    print()
    print(optimizer.report())


if __name__ == "__main__":
    main()
