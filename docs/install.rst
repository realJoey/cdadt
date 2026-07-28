Installation
============

cdadt is developed and tested in a conda environment named ``cdadt_env``, on Windows, with
Python 3.11. Several of the choices below are not stylistic; each one is explained, because an
environment that is nearly right produces a stack that imports and then aborts the interpreter
in the middle of a Newton solve.

The environment
---------------

.. code-block:: bash

   conda create -y -n cdadt_env -c conda-forge python=3.11 "numpy<2" scipy matplotlib pyyaml \
       openmdao pyoptsparse ipopt cyipopt pytest pytest-cov sphinx sphinx_rtd_theme ruff black
   conda install -y -n cdadt_env -c conda-forge "libblas=*=*openblas"
   conda activate cdadt_env

Then install OpenConcept and cdadt, in that order:

.. code-block:: bash

   pip install -e /path/to/openconcept --no-deps
   pip install -e ".[dev,docs]"

Why each of those flags is there
--------------------------------

``numpy<2``
    OpenConcept 1.2.6 declares an upper bound on NumPy. Installing NumPy 2 produces a stack
    that imports and then fails inside components that use APIs removed in 2.0.

``--no-deps`` on OpenConcept
    Without it, pip reads OpenConcept's own requirements and *acts* on them, downgrading the
    conda-forge NumPy, SciPy and OpenMDAO that conda just resolved. The result is a mixed
    conda/pip stack that is difficult to diagnose because nothing about it looks wrong. cdadt
    deliberately does not list OpenConcept in its own dependencies for the same reason.

The OpenBLAS pin
    Not optional on Windows. With an MKL-backed BLAS, NumPy aborts the interpreter with
    ``0xc06d007f`` inside ``numpy.linalg.solve`` -- which OpenConcept calls at import time, so
    the failure happens before any cdadt code runs. Forcing the OpenBLAS build of ``libblas``
    avoids it.

``pyoptsparse``, ``ipopt`` and ``cyipopt``
    Optional for sizing, effectively required for optimization. The reason is specific: the
    black box is a Newton-solved implicit system, and an optimizer exploring a wide design
    space will eventually propose a design it cannot converge. OpenMDAO's pyOptSparse driver
    catches that, reports the point as failed, and lets the optimizer shorten its step. SciPy's
    SLSQP driver re-raises, so one unconvergeable trial design ends the run. SLSQP is supported
    and is the code's default because it is always available; the shipped optimization case
    uses IPOPT.

Reproducing the exact environment
---------------------------------

``environment.yml`` at the repository root is an export of the environment these results were
produced in, and can be used directly:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate cdadt_env
   pip install -e /path/to/openconcept --no-deps
   pip install -e ".[dev,docs]"

Which OpenConcept
-----------------

cdadt drives whichever OpenConcept is importable; it is loaded by name from the case file. The
numbers in this documentation were produced against the clone described in :doc:`validation`,
which records the commit and the one respect in which it differs from OpenConcept's published
``main``. That matters for reproducibility and is stated there rather than assumed.

Checking the installation
-------------------------

.. code-block:: bash

   pytest -q -m "not slow"     # the fast loop
   pytest -q                   # everything, including the live reference comparison
   cdadt inspect cases/b738.yaml

The full suite runs the shipped sizing case, OpenConcept's own example, and the shipped
optimization study. If it passes, the installation is not merely importable but produces the
documented numbers.
