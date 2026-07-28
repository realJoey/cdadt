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

   conda create -y -n cdadt_env -c conda-forge python=3.11 numpy scipy matplotlib \
       openmdao pyoptsparse ipopt cyipopt pytest pytest-cov pyyaml sphinx numpydoc
   conda install -y -n cdadt_env -c conda-forge "libblas=*=*openblas"
   conda activate cdadt_env
   pip install sphinx_mdolab_theme

Everything except the documentation theme comes from ``conda-forge``. This matters most for
``pyoptsparse``, ``ipopt``, and ``cyipopt``: conda-forge ships prebuilt IPOPT binaries for
Windows, which avoids compiling the optimizer from source.

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
   OpenConcept's ``setup.py`` declares ``numpy >=1.20, <2``. Without ``--no-deps`` pip acts
   on that bound and downgrades NumPy underneath the conda-forge stack, which breaks the
   compiled ``pyoptsparse``/IPOPT extensions that were built against the conda NumPy ABI.
   cdadt's own dependencies are installed by conda beforehand, so there is nothing for pip
   to resolve.

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
