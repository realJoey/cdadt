# cdadt

A certification-driven aircraft design tool.

cdadt sizes and optimizes an aircraft against an explicit certification basis. Full mission
sizing — balanced-field takeoff, climb, cruise, descent, 14 CFR Part 25 reserves and loiter —
comes from [OpenConcept](https://github.com/mdolab/openconcept), used as a black box: cdadt
sets its inputs, converges it, and reads its outputs. **No OpenConcept source is modified, and
no OpenConcept class is subclassed.** Both are enforced by the test suite.

## Validated against the reference

cdadt reproduces OpenConcept's own `B738_sizing.py` example, every quantity, at 21 nodes per
phase:

| Quantity | OpenConcept | cdadt |
|---|---|---|
| MTOW (kg) | 78345.6435 | 78345.6435 |
| OEW (kg) | 41748.3258 | 41748.3258 |
| Block fuel (kg) | 15977.0628 | 15977.0628 |
| Total fuel with reserves (kg) | 18597.3177 | 18597.3177 |
| Balanced field length (ft) | 5247.7948 | 5247.7948 |
| Horizontal tail area (m²) | 27.9332 | 27.9332 |
| Vertical tail area (m²) | 20.2101 | 20.2101 |

Agreement is better than 1e-6 relative — the Newton solver's own convergence. The reference is
*run* in the test, not quoted from a table.

## Optimized against a certification basis

Minimizing total mission fuel over wing area, aspect ratio, sweep, taper and engine rating,
subject to §25.113 field length, §25.121(b) engine-out climb gradient, approach category and
throttle limits:

| Quantity | Baseline | Optimum | Change |
|---|---|---|---|
| Total fuel (kg) | 18596.83 | 15991.39 | **−14.0%** |
| MTOW (kg) | 78345.02 | 71959.34 | −8.2% |
| Engine rating (lbf) | 27000 | 21911 | −18.8% |

Five of five requirements met, one active — the climb throttle limit. That constraint is what
shaped the design.

```
regulation             requirement                                 value        limit       margin units  status
----------------------------------------------------------------------------------------------------------------
design                 Throttle within limit in climb             1.0000       1.0000      -0.0000 -      ACTIVE
14 CFR 25.125 / 97     Approach speed within category           136.4991     140.0000       3.5009 kn     MET
design                 Throttle within limit in cruise            0.8287       1.0000       0.1713 -      MET
14 CFR 25.113          Takeoff distance within field availa    6263.7145    8000.0000    1736.2855 ft     MET
14 CFR 25.121(b)       OEI second-segment climb gradient          0.0406       0.0240       0.0166 rad    MET
```

## Design rules

- **Every discipline is a class.** Aerodynamics, propulsion, mass, geometry, stability, high
  lift and weights are `om.Group` subclasses that own their own analysis, promote their own
  variables, and never reach into a sibling. No global state, no module-level caches, no
  free-function discipline math.
- **OpenConcept is composed, never inherited.** Subclassing an OpenConcept component would
  change its behaviour while leaving its source untouched — a modification no diff would show.
  A test walks every class cdadt defines and fails on any OpenConcept base.
- **Nothing is defaulted that belongs to a case.** Every certification limit is a constructor
  argument, and so is its `source`. A limit with no recorded provenance is indistinguishable
  from a guess in the report that quotes it.
- **What is not validated is stated as prominently as what is.** See `docs/validation.rst`.

## Install

```bash
conda create -y -n cdadt_env -c conda-forge python=3.11 "numpy<2" scipy matplotlib pyyaml \
    openmdao pyoptsparse ipopt cyipopt pytest pytest-cov sphinx sphinx_rtd_theme ruff black
conda install -y -n cdadt_env -c conda-forge "libblas=*=*openblas"
conda activate cdadt_env

pip install -e /path/to/openconcept --no-deps   # --no-deps is required; see docs/install.rst
pip install -e ".[dev]"
```

The OpenBLAS pin is not optional on Windows: with MKL-backed BLAS, NumPy aborts the
interpreter (`0xc06d007f`) inside `numpy.linalg.solve`, which OpenConcept calls at import.

## Run

```bash
python examples/size_b738.py                          # size, print results
python examples/optimize_b738.py                      # optimize against the certification basis
python examples/optimize_b738.py --objective MTOW

pytest -q -m "not slow"                               # 43 tests, ~1 s
pytest -q                                             # everything, ~13 s
cd docs && make html
```

## Layout

```
cdadt/
  aircraft.py       AircraftDefinition — ac| design parameters as encapsulated state
  mission.py        MissionProfile, PhaseSchedule — the mission and its continuation schedule
  disciplines/      one class per engineering domain
  model.py          JetTransportPhaseModel (what OpenConcept builds per phase), SizingModel
  sizing.py         SizingAnalysis — build, converge, read results
  certification.py  Requirement, CertificationBasis, and the shipped Part 25 requirements
  optimization.py   DesignOptimizer, DesignVariable
cases/              b738_aircraft.yaml, b738_mission.yaml
examples/           size_b738.py, optimize_b738.py
tests/              unit / integration / contract / validation
docs/               Sphinx
```

## Documentation

| Page | Contents |
|---|---|
| `install.rst` | Environment setup, and why each flag is required |
| `tutorials.rst` | Size, change the aircraft, optimize, add a requirement, swap a discipline |
| `architecture.rst` | The eight classes, the two discipline scopes, the sizing loop |
| `blackbox.rst` | Exactly what "black box" means, what goes in, what comes out |
| `mission.rst` | The mission profile and why continuation is part of the interface |
| `certification.rst` | Requirements, scaling, the traceability matrix, what is not modeled |
| `optimization.rst` | Design variables, order of operations, why scaling decides the result |
| `validation.rst` | What is validated, against what, **and what is not** |

## License

MIT. See `LICENSE`.
