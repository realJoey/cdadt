# cdadt

A certification-driven aircraft design tool.

cdadt sizes and optimizes an aircraft against an explicit certification basis. The sizing
itself — balanced-field takeoff, climb, cruise, descent, 14 CFR Part 25 reserves and loiter —
is performed by an [OpenConcept](https://github.com/mdolab/openconcept) analysis used as a
**black box**: cdadt sets its inputs, converges it, and reads its outputs.

```bash
cdadt size     cases/b738.yaml
cdadt optimize cases/b738_optimization.yaml
cdadt inspect  cases/b738.yaml
```

## Three claims, all enforced by the test suite

**OpenConcept is a black box.** No cdadt module imports OpenConcept at all. The sizing analysis
is named in the case file as `module:ClassName` and loaded at run time, so there is no
OpenConcept class cdadt can subclass, no component it can re-wire, and no physics it can quietly
reimplement. A contract test parses every cdadt source file to prove it, another walks every
class for an OpenConcept base, and a third checks the OpenConcept working tree is untouched.

**Every discipline is a class.** Geometry, aerodynamics, propulsion, stability, structures,
weights and performance are classes with encapsulated state. Each owns exactly one slice of the
black box's interface — which variables its domain sets, which responses its domain reports —
and ownership is checked to be total and disjoint against the *live* model, all 241 settable
variables of it. No global state, no module-level cache, no free function doing discipline work.

**A study is a file.** Every design variable, everything written into the box before it is
converged, the continuation ladder, the driver, the objective and the certification constraints
are all declared in one YAML case file, laid out block for block like OpenConcept's own `B738.py`
run script. A variable becomes free for the optimizer by gaining an `optimize:` entry where it is
already declared, so two studies that ask different questions of the same aeroplane differ only
in data — and never disagree about a number.

## The black box is the only one it could have been

`B738SizingMissionAnalysis` is the **sole** analysis in OpenConcept's ~30,000 lines that combines
a balanced-field takeoff, Part 25 reserves, a closed weight loop and a scalable engine. Every
other example is fixed-weight, or lacks the takeoff, or lacks the reserves. That is a survey
result, not a preference — see `docs/openconcept.rst`.

## Verified: the equations are solved right

| Study | Result |
|---|---|
| **Grid convergence** | Observed order **4.7** (Simpson's rule is 4th order); shipped 21-node grid converged to **1e-6** |
| **Total derivatives** | Agree with finite differences to **1.1e-4**, with the textbook truncation/round-off minimum at step 1e-6 |
| **Solver tolerance** | Every tolerance probed down to **1e-12** is reachable on every grid, so the shipped 1e-9 has three decades of margin |
| **Reproducibility** | Three different continuation ladders reach the same aircraft to **1e-7**; reruns are bit-identical |
| **Environment** | Rebuilt from scratch out of `environment.yml`; the suite passes and every number reproduces |
| **Optimality** | No feasible ±2% perturbation of any design variable improves the objective |
| **Coverage** | **100%** of statements and branches, enforced, no exclusion list |

Details and the full tables in `docs/verification.rst`.

## Validated against the reference, run rather than quoted

`tests/test_validation.py` imports OpenConcept's own `run_738_sizing_analysis`, runs it in the
same process at the same grid, and compares every quantity at 1e-6 relative — the Newton
solver's own convergence, not an engineering tolerance.

| Quantity | Value |
|---|---|
| Maximum takeoff weight | 78,345.6435 kg |
| Operating empty weight | 41,748.3258 kg |
| Block fuel | 15,977.0628 kg |
| Fuel with reserves | 18,597.3177 kg |
| Balanced field length | 5,247.7948 ft |
| Horizontal tail area | 27.9332 m² |
| Vertical tail area | 20.2101 m² |

And validated against the real aeroplane, independently of OpenConcept: MTOW within **0.8%** and
OEW within **0.8%** of published 737-800 figures, with cruise L/D 17.5, TSFC 0.614 lb/lbf/hr,
wing loading 629 kg/m² and thrust-to-weight 0.313 — every dimensionless group inside the band a
narrow-body transport occupies.

## Optimized against a certification basis

Minimizing fuel with reserves over the wing planform and the engine rating, subject to 14 CFR
25.113, 25.121(b)(1)(i) and the engine deck's throttle band:

| Quantity | Baseline | Optimum | Change |
|---|---|---|---|
| Fuel with reserves (kg) | 18,596.8 | 15,914.6 | **−14.4%** |
| Maximum takeoff weight (kg) | 78,345.0 | 71,340.0 | −8.9% |
| Engine rating (lbf) | 27,000 | 20,774.5 | −23.1% |

Four of four constraints met, one active — the climb throttle band. Neither certification
constraint binds: the runway keeps 1,413 ft of margin and the second-segment gradient more than
double its minimum. Three of the five design variables end on a bound, which is the bounds doing
the modelling and is called out as such in `docs/optimization.rst`.

```
regulation              constraint                                             value        bound      margin  units  status
---------------------------------------------------------------------------------------------------------------------------
14 CFR 25.113           Balanced field length within the runway available  6587.1294      <= 8000  1412.8706   ft     MET
14 CFR 25.121(b)(1)(i)  OEI second-segment climb gradient                     0.0501     >= 0.024     0.0261   rad    MET
-                       Climb throttle within the engine deck                 1.0500 0.01 to 1.05     0.0000   -      ACTIVE
-                       Cruise throttle within the engine deck                0.8691 0.01 to 1.05     0.1809   -      MET

Design constraints with no stated regulation or source: climb_throttle, cruise_throttle.
These bound the design; they are not certification evidence.
```

**Read `docs/validation.rst` before quoting any of this.** It states what has been established
and, at least as prominently, what has not — including that §25.121(b) is evaluated clean rather
than in the takeoff configuration the regulation specifies, that there is no V<sub>MC</sub>, no
approach speed and no CG model, and which OpenConcept commit these numbers correspond to.

The installed clone is exactly upstream `mdolab/openconcept`. Nothing in this repository modifies OpenConcept, and nothing is ever pushed to it. The installed
clone is checked for uncommitted changes by the test suite, and any commit it carries that
upstream does not is checked against the modules cdadt actually loads.

## Install

```bash
conda create -y -n cdadt_env -c conda-forge python=3.11 "numpy<2" scipy matplotlib pyyaml \
    openmdao pyoptsparse ipopt cyipopt pytest pytest-cov sphinx sphinx_rtd_theme ruff black
conda install -y -n cdadt_env -c conda-forge "libblas=*=*openblas"
conda activate cdadt_env

pip install -e /path/to/openconcept --no-deps   # --no-deps is required; see docs/install.rst
pip install -e ".[dev,docs]"
```

The OpenBLAS pin is not optional on Windows: with MKL-backed BLAS, NumPy aborts the interpreter
(`0xc06d007f`) inside `numpy.linalg.solve`, which OpenConcept calls at import.

## Run

```bash
cdadt size cases/b738.yaml                        # ~5 s at 21 nodes per phase
cdadt optimize cases/b738_optimization.yaml       # ~2 min
cdadt inspect cases/b738.yaml --what inputs       # what the box accepts

cdadt size cases/b738.yaml --outputs b738_out     # + n2.html, trajectory.pdf, report.txt, results.json

pytest -q -m "not slow"                           # the fast loop, 216 tests, ~2 min
pytest -q                                         # 272 tests, ~8 min
pytest -q --cov=cdadt                             # and 100% statement + branch coverage
pytest -q -m verification                         # grid, derivatives, solver, reproducibility
pytest -q -m validation                           # reference match + physical checks
cd docs && make html
```

Tests are marked by the class of claim they make — `unit` (143), `contract` (14),
`integration` (35), `verification` (39), `validation` (19) — so "the suite passes" is a
statement about what has been established, not one undifferentiated green tick.

Coverage is **100% of statements and branches**, enforced by `fail_under = 100`, with no
exclusion list. That is a fair target here only because cdadt computes no physics: everything in
it is interface, routing, validation and reporting, so an unreachable line is dead code or a
missing test.

## Layout

```
cdadt/
  parameters.py     Parameter, Response — the two value objects
  disciplines/      one class per engineering domain, each owning its slice of the interface
  aircraft.py       Aircraft — composes the disciplines, routes every ac| name to one owner
  mission.py        InitialConditions, ContinuationLadder, ContinuationStep
  blackbox.py       OpenConceptSizingBox — loaded by name, set, converged, read
  config.py         the case file, validated hard
  analysis.py       SizingAnalysis — build, converge, read
  results.py        ResponseCatalog, SizingResults
  certification.py  Constraint, ConstraintResult, CertificationBasis, the traceability matrix
  optimization.py   Optimizer, OptimizationOutcome
  artifacts.py      StudyArtifacts, MissionTrajectory — the N2, the trajectory plot, the report
  cli.py            cdadt size | optimize | inspect
cases/              b738.yaml, b738_optimization.yaml
examples/           size_b738.py, optimize_b738.py
tests/              unit / contract / integration / validation
docs/               Sphinx
```

## Documentation

| Page | Contents |
|---|---|
| `install.rst` | Environment setup, and why each flag is required |
| `quickstart.rst` | The three commands and what they print |
| `tutorials.rst` | Change the aircraft, the mission, the question; free a variable, add a constraint or a discipline |
| `architecture.rst` | The classes, what each owns, and the tests that enforce the rules |
| `blackbox.rst` | Exactly what "black box" means, what goes in, what comes out, what is fixed inside |
| `configuration.rst` | Every key of the case file |
| `mission.rst` | The initial conditions, and why the continuation ladder is part of the interface |
| `certification.rst` | Constraints, provenance, the traceability matrix, what cannot be constrained |
| `optimization.rst` | Design variables, scaling, driver choice, and why IPOPT |
| `artifacts.rst` | The files `--outputs` writes, and what is plotted |
| `verification.rst` | Grid convergence, derivative accuracy, solver tolerance, reproducibility, optimality |
| `validation.rst` | What is validated, against what, **and what is not** |
| `openconcept.rst` | The survey of all ~30,000 lines, and why this black box is the only candidate |
| `interface.rst` | The generated input/output reference |

## License

MIT. See `LICENSE`.
