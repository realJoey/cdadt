.. _install:

************
Installation
************

cdadt is developed against a dedicated conda environment named ``cdadt_env``. The recipe
below is the supported configuration; the environment is also captured exactly in
``environment.yml`` at the repository root.

The environment
===============

.. code-block:: bash

   conda create -y -n cdadt_env -c conda-forge python=3.11 "numpy<2" scipy matplotlib \
       openmdao pyoptsparse ipopt cyipopt pytest pytest-cov pyyaml sphinx numpydoc \
       packaging black ruff "libblas=*=*openblas"
   conda activate cdadt_env
   pip install sphinx_mdolab_theme

Everything except the documentation theme comes from ``conda-forge``. This matters most for
``pyoptsparse``, ``ipopt``, and ``cyipopt``: conda-forge ships prebuilt IPOPT binaries for
Windows, which avoids compiling the optimizer from source.

The stack this resolves to, and the one cdadt's results were produced with:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Note
   * - Python
     - 3.11.15
     - OpenConcept's CI upper bound
   * - NumPy
     - 1.26.4
     - Required: OpenConcept declares ``<2``, and the bound is real
   * - OpenMDAO
     - 3.41.0
     - What conda selects under the NumPy bound
   * - pyOptSparse / IPOPT
     - 2.16.0 / 3.14.19
     - Prebuilt, works against NumPy 1.26

.. note::
   OpenMDAO 3.41 is noticeably slower than 3.43 on the coupled sizing optimization -- the
   IPOPT run takes tens of minutes rather than one. That is a cost of respecting
   OpenConcept's NumPy bound, and it is worth paying: under NumPy 2 the reference does not
   converge at all, so nothing built on it can be checked.

.. warning::
   **Pin the BLAS implementation to OpenBLAS.** The second ``conda install`` line above is
   not optional on Windows. Solving the environment without it selects the MKL-backed
   ``libblas``/``libcblas``/``liblapack`` variants, and the resulting NumPy aborts the
   interpreter with a Windows fatal exception (``0xc06d007f``) the first time
   ``numpy.linalg.solve`` is called. OpenConcept calls it at import time, in
   ``openconcept/atmospherics/atmospherics_data.py``, so the failure appears as an
   unrecoverable crash while importing ``openconcept.mission`` rather than as a Python
   traceback.

   For the same reason, do not mix a pip-installed NumPy or SciPy into this environment.
   PyPI wheels carry their own OpenBLAS, and loading them alongside the conda BLAS
   produces the same class of crash. Every array package here must come from conda-forge.

.. note::
   ``sphinx_mdolab_theme`` has no conda-forge package and is installed with pip. It is a
   documentation dependency only; nothing in ``cdadt`` imports it.

Installing OpenConcept
======================

cdadt's mission analysis is supplied by OpenConcept, which is installed **editable from a
local clone and without dependency resolution**:

.. code-block:: bash

   pip install -e /path/to/openconcept --no-deps

Both flags are deliberate.

``--no-deps``
   cdadt's dependencies are installed by conda beforehand, so there is nothing for pip to
   resolve, and letting pip re-resolve them can replace conda-forge builds with PyPI wheels
   that carry their own BLAS.

   **This flag does not license ignoring OpenConcept's version bounds.** It declares
   ``numpy >=1.20, <2``, and the conda environment above satisfies that bound deliberately.
   An earlier version of this project installed NumPy 2 with ``--no-deps`` and then spent
   considerable effort concluding that OpenConcept had a defect, because its own B738 test
   failed. It did not: the environment was wrong. Under NumPy 2, OpenConcept's parasite drag
   buildup evaluates ``log(0.06 Re)`` and ``sqrt(Re)`` at the first nodes of its ground roll
   and the solver reaches ``NaN``; under NumPy 1.26 the same test passes.

``-e`` from a local clone
   The editable install points Python at the working clone rather than a copy, so any
   local compatibility patches in that clone remain in effect and the exact OpenConcept
   revision used for a result is recoverable from its git history.

Installing cdadt
================

.. code-block:: bash

   cd /path/to/cdadt
   pip install -e ".[dev]"

Verifying the installation
==========================

The environment is not considered working until its gate tests pass:

.. code-block:: bash

   pytest cdadt/tests/test_environment.py cdadt/tests/test_openconcept_integrity.py -v

The strongest single check is OpenConcept's own test suite. If the dependency cannot pass
its own tests, nothing built on it can be trusted:

.. code-block:: bash

   pytest ../openconcept/openconcept/examples/tests/test_example_aircraft.py -k B738Sizing -v

These tests do more than check imports. They

* build a real ``FullMissionWithReserve`` mission group,
* ask IPOPT to solve an actual constrained problem and check that it returns the analytic
  optimum,
* confirm OpenConcept resolves to a git working tree (proving the editable install took
  effect rather than a PyPI wheel shadowing it), and
* confirm the OpenConcept clone has no uncommitted modifications.

The last point is a standing rule, not a nicety: **cdadt never modifies OpenConcept.** Any
behavioral difference cdadt needs is expressed in cdadt by subclassing, wrapping, or adding
a new component. The test suite enforces this both on disk (``git status`` must be clean)
and in source (an AST scan rejects any assignment onto an OpenConcept symbol), so the rule
cannot erode silently.

Running the test suite
======================

.. code-block:: bash

   pytest -q -m "not slow"    # fast development loop
   pytest -q                  # full suite, including reference validation runs

Test markers correspond to the classes of claim described in :ref:`verification`.
