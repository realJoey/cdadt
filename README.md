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

**A study is a file.** The aircraft, the mission, the continuation ladder, the design variables,
the objective and the certification requirements are all declared in one YAML case file. Two
studies that ask different questions of the same aeroplane differ only in data.

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

## Optimized against a certification basis

Minimizing fuel with reserves over the wing planform and the engine rating, subject to 14 CFR
25.113, 25.121(b)(1)(i) and the engine deck's throttle limits:

| Quantity | Baseline | Optimum | Change |
|---|---|---|---|
| Fuel with reserves (kg) | 18,596.8 | 15,991.4 | **−14.0%** |
| Maximum takeoff weight (kg) | 78,345.0 | 71,959.3 | −8.2% |
| Engine rating (lbf) | 27,000 | 21,911 | −18.9% |

Four of four requirements met, one active — the climb throttle limit. That is the constraint
that shaped the design: not the runway, and not the second-segment climb gradient, both of which
keep large margins.

```
regulation               requirement                                          value       limit      margin  units  status
14 CFR 25.113            Balanced field length within the runway available  6263.7147  8000.0000  1736.2853  ft     MET
14 CFR 25.121(b)(1)(i)   OEI second-segment climb gradient                     0.0554     0.0240     0.0314  rad    MET
design                   Throttle within the engine deck's range in climb      1.0000     1.0000    -0.0000  -      ACTIVE
design                   Throttle within the engine deck's range in cruise     0.8287     1.0000     0.1713  -      MET
```

**Read `docs/validation.rst` before quoting any of this.** It states what has been established
and, at least as prominently, what has not — including that §25.121(b) is evaluated clean rather
than in the takeoff configuration the regulation specifies, that there is no V<sub>MC</sub>, no
approach speed and no CG model, and which OpenConcept commit these numbers correspond to.

Nothing in this repository modifies OpenConcept, and nothing is ever pushed to it. The installed
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

pytest -q -m "not slow"                           # the fast loop
pytest -q                                         # 128 tests, ~1 min
cd docs && make html
```

## Layout

```
cdadt/
  parameters.py     Parameter, Response — the two value objects
  disciplines/      one class per engineering domain, each owning its slice of the interface
  aircraft.py       Aircraft — composes the disciplines, routes every ac| name to one owner
  mission.py        MissionProfile, PhaseSchedule, ContinuationStep
  blackbox.py       OpenConceptSizingBox — loaded by name, set, converged, read
  config.py         the case file, validated hard
  analysis.py       SizingAnalysis — build, converge, read
  results.py        ResponseCatalog, SizingResults
  certification.py  Requirement, CertificationBasis, the traceability matrix
  optimization.py   Optimizer, OptimizationOutcome
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
| `tutorials.rst` | Change the aircraft, the mission, the question; add a requirement or a discipline |
| `architecture.rst` | The classes, what each owns, and the tests that enforce the rules |
| `blackbox.rst` | Exactly what "black box" means, what goes in, what comes out, what is fixed inside |
| `configuration.rst` | Every key of the case file |
| `mission.rst` | The profile, and why the continuation ladder is part of the interface |
| `certification.rst` | Requirements, provenance, the traceability matrix, what cannot be constrained |
| `optimization.rst` | Design variables, scaling, driver choice, and why IPOPT |
| `interface.rst` | The generated input/output reference |
| `validation.rst` | What is validated, against what, **and what is not** |

## License

MIT. See `LICENSE`.
