Installation
============

cdadt runs in a conda environment called ``cdadt_env``. Two of the steps below look like
superstition and are not; each has a failure it prevents, stated with it.

The environment
---------------

.. code-block:: bash

   conda create -y -n cdadt_env -c conda-forge python=3.11 \
       "numpy<2" scipy matplotlib pyyaml \
       openmdao pyoptsparse ipopt cyipopt \
       pytest pytest-cov sphinx numpydoc sphinx_rtd_theme ruff black

   conda install -y -n cdadt_env -c conda-forge "libblas=*=*openblas"
   conda activate cdadt_env

The packages
------------

.. code-block:: bash

   # OpenConcept, from a clone, with --no-deps
   pip install -e /path/to/openconcept --no-deps

   # cdadt itself
   pip install -e ".[dev]"

Verify:

.. code-block:: bash

   pytest -q -m "not slow"    # 43 tests, about a second
   pytest -q                  # everything, including the live OpenConcept comparison

Why ``--no-deps``
-----------------

OpenConcept declares ``numpy>=1.20,<2``. Installing it without ``--no-deps`` lets pip act on
that bound and replace the conda-forge NumPy with a pip wheel, which leaves the environment
with two NumPy builds and one BLAS that matches neither.

The bound itself is real: under NumPy 2, OpenConcept's own B738 test fails. Build the
environment with ``numpy<2`` and let conda, not pip, be the one that installs it.

Why the OpenBLAS pin
--------------------

On Windows, with MKL-backed BLAS, NumPy aborts the interpreter with ``0xc06d007f`` inside
``numpy.linalg.solve`` -- which OpenConcept calls while building its engine surrogates, so the
failure happens at import and looks like a broken install rather than a BLAS conflict. Pinning
``libblas=*=*openblas`` avoids it. On Linux and macOS the pin is harmless.

Reproducing the environment exactly
-----------------------------------

``environment.yml`` in the repository root is a full export of a working environment,
including build strings:

.. code-block:: bash

   conda env create -f environment.yml

When a result disagrees with the reference, suspect the environment before suspecting the
reference.

The OpenConcept clone
---------------------

cdadt is developed against a local clone of OpenConcept installed in editable mode. The clone
must stay clean: :mod:`tests.test_openconcept_boundary` fails the suite if ``git status`` in
the OpenConcept working tree reports uncommitted modifications. That test is the enforcement
behind the claim in :doc:`blackbox`, and it is the reason the claim can be made at all.
