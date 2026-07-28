# cdadt

A Certification Driven Aircraft Design Tool.

cdadt sizes and optimizes aircraft against an explicit certification basis. Regulatory
requirements — 14 CFR Part 25 §25.113 balanced field length, §25.121 one-engine-inoperative
climb gradient, §25.125 landing field length, approach speed category, and thrust margin —
are modeled as first-class objects that know their citation, the variables they read, and
the optimizer constraint they register. Every run emits a traceability matrix linking each
regulation to the constraint that enforced it and the margin achieved.

## Result

Minimizing total mission fuel for a Boeing 737-800 — design mission plus Part 25 reserves,
2800 nmi with 18 t payload — with wing area, aspect ratio, sweep, taper and engine rating
free. IPOPT exits `Solve Succeeded`:

| Quantity | Baseline | Optimum | Change |
|---|---|---|---|
| Total fuel (kg) | 18,594 | 17,164 | **−7.7%** |
| MTOW (kg) | 78,341 | 75,249 | −3.9% |
| Engine rating (lbf) | 27,000 | 22,185 | −17.8% |

Ten of ten requirements met, **two active** — the category C approach speed limit and the
climb throttle limit. Those two shaped the design.

```
regulation           requirement                          value       limit      margin units  status
-----------------------------------------------------------------------------------------------------
14 CFR 97 / ICAO cat Approach speed within category    140.0000    140.0000     -0.0000 kn     ACTIVE
design               Throttle within limit in climb      1.0500      1.0500      0.0000 -      ACTIVE
14 CFR 25.125        Landing field length within a    6395.0866   7000.0000    604.9134 ft     MET
14 CFR 25.121(b)     OEI second-segment climb grad       0.0311      0.0240      0.0071 rad    MET
14 CFR 25.113        Takeoff distance within field    5213.5158   8000.0000   2786.4842 ft     MET
```

## Design rules

- **Every discipline is an object.** Aerodynamics, propulsion, weights, geometry, stability
  and landing are classes with encapsulated state that declare what they provide and
  require. No global state, no free-function discipline math. Each delegates to a swappable
  *provider*, so an empirical buildup and a high-fidelity analysis satisfy one interface.
- **The mission analysis is a black box.** Full mission sizing — balanced-field takeoff,
  climb, cruise, descent, Part 25 reserves, loiter — comes from
  [OpenConcept](https://github.com/mdolab/openconcept). cdadt consumes it through a single
  boundary class and **never modifies OpenConcept**. An AST scan fails the test suite if any
  other module imports it, and a `git status` check fails if the clone is dirty.
- **Nothing is defaulted.** `AircraftConfiguration.value` has no `default` parameter and
  will not get one. A method constant supplied as a default is a hardcoded number that
  happens to carry a citation: it applies itself to configurations that never mentioned it.
  This includes constants OpenConcept itself defaults — tail volume coefficients, the MLW
  fraction, the engine deck.
- **Nothing is claimed that is not verified.** Every test states its class of claim (unit /
  contract / derivative / integration / validation / regression), and `docs/validation.rst`
  states what has *not* been validated with equal prominence.

## Installation

Requires the `cdadt_env` conda environment. Full instructions, including the mandatory
OpenBLAS pin and the `--no-deps` OpenConcept install, are in `docs/install.rst`.

```bash
conda create -y -n cdadt_env -c conda-forge python=3.11 numpy scipy matplotlib \
    openmdao pyoptsparse ipopt cyipopt pytest pytest-cov pyyaml sphinx numpydoc
conda install -y -n cdadt_env -c conda-forge "libblas=*=*openblas"
conda activate cdadt_env
pip install -e /path/to/openconcept --no-deps
pip install -e ".[dev]"
pytest cdadt/tests/test_environment.py -v
```

> The OpenBLAS pin is not optional on Windows. With MKL-backed BLAS, NumPy aborts the
> interpreter (`0xc06d007f`) inside `numpy.linalg.solve`, which OpenConcept calls at import.

## Running

```bash
python examples/optimize_b738.py                      # size, optimize, report
python examples/optimize_b738.py --objective MTOW     # minimize takeoff weight instead
pytest -q -m "not slow"                               # fast test loop
pytest -q                                             # everything (~70 s)
```

## Documentation

```bash
cd docs && make html
```

| Page | Contents |
|------|----------|
| `install.rst` | Environment setup and why each flag is required |
| `tutorials.rst` | Size, optimize, add a provider, add a requirement |
| `architecture.rst` | OOP contracts, disciplines, providers, coupling checks |
| `blackbox.rst` | The OpenConcept contract and why it is a contract, not a wall |
| `certification.rst` | Requirements, the traceability matrix, how wiring is verified |
| `optimization.rst` | The design problem, drivers, and why scaling decided the result |
| `verification.rst` | Classes of claim and the rules the test suite obeys |
| `validation.rst` | What is validated, against what, **and what is not** |

## Validation

cdadt reproduces OpenConcept's own published golden values for the B737-800:

| Quantity | OpenConcept golden | cdadt | Difference |
|---|---|---|---|
| Block fuel | 35,213.767 lbm | 35,215.7 lbm | +0.006% |
| Total fuel | 40,991.188 lbm | 40,991.9 lbm | +0.002% |
| MTOW | 172,711.303 lbm | 172,711.4 lbm | +0.0002% |

A live run of the reference example is also compared quantity by quantity — balanced field
length, V₁, tail areas, empty and landing weights. One quantity **deliberately** differs: the
engine-out climb gradient. OpenConcept's example evaluates it with clean drag, while
§25.121(b) specifies takeoff flaps; cdadt deploys them and gets 0.0422 rad against the
reference's 0.0579. See `docs/validation.rst`.

> **Environment matters.** OpenConcept declares `numpy >=1.20, <2` and the bound is real —
> under NumPy 2 its own B738 test fails. Build the environment as documented; when the
> reference disagrees with cdadt, the first suspect is the environment, not the reference.

## License

MIT. See `LICENSE`.
