# cdadt

A Certification Driven Aircraft Design Tool.

cdadt sizes and optimizes aircraft against an explicit certification basis. Regulatory
requirements — 14 CFR Part 25 §25.113 balanced field length, §25.121 one-engine-inoperative
climb gradients, §25.119 and §25.125 landing performance, and thrust/throttle margins — are
modeled as first-class objects that know their citation, the variables they read, and the
optimizer constraint they register. A run emits a traceability matrix linking every
regulation to the constraint that enforced it and the margin achieved.

## Design rules

- **Every discipline is an object.** Aerodynamics, propulsion, weights, geometry, and
  stability are classes with encapsulated state that declare what they provide and require.
  No global state, no free-function discipline math. Each delegates its physics to a
  swappable *provider*, so an empirical buildup and a high-fidelity analysis satisfy the
  same interface.
- **The mission analysis is a black box.** Full mission sizing — balanced-field takeoff,
  climb, cruise, descent, Part 25 reserves, and loiter — comes from
  [OpenConcept](https://github.com/mdolab/openconcept). cdadt consumes it through a single
  boundary class and **never modifies OpenConcept**. The test suite enforces this.
- **Nothing is claimed that is not verified.** Every test states which class of claim it
  makes (unit / contract / derivative / integration / validation / regression). Results are
  validated against the OpenConcept reference implementation and published hand
  calculations.

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
```

Verify the install:

```bash
pytest cdadt/tests/test_environment.py cdadt/tests/test_openconcept_integrity.py -v
```

## Documentation

Built with Sphinx from `docs/`:

```bash
cd docs && make html
```

| Page | Contents |
|------|----------|
| `docs/install.rst` | Environment setup and why each install flag is required |
| `docs/verification.rst` | V&V strategy, classes of claim, rules the test suite obeys |

## License

MIT. See `LICENSE`.
