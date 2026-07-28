"""Environment gate for cdadt_env.

Claim class: ``unit``. These tests prove the running interpreter has the toolchain cdadt
depends on and that the dependency is wired the way ``docs/install.rst`` says it is. They
prove nothing about cdadt physics.

Every assertion here exercises the real dependency: IPOPT is asked to solve an actual
constrained problem rather than merely be importable, and OpenConcept's mission group is
built rather than merely imported.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt.tests.support import (
    available_pyoptsparse_optimizers,
    openconcept_package_dir,
    openconcept_repo_root,
)

pytestmark = pytest.mark.unit


def test_openmdao_available_at_required_version():
    """OpenMDAO imports and meets the minimum version cdadt's mission machinery needs.

    The version lives on the ``openmdao`` package, not on ``openmdao.api``, which only
    re-exports the public symbols.
    """
    import openmdao
    from packaging.version import Version

    assert Version(openmdao.__version__) >= Version(
        "3.21"
    ), f"cdadt requires openmdao>=3.21 (OpenConcept's own floor); found {openmdao.__version__}"


def test_openconcept_importable_and_mission_builds():
    """OpenConcept is importable and its full mission group actually builds.

    Importing is not enough: the mission group instantiates a user-supplied aircraft
    model class inside every phase, so a build is the first point at which an
    incompatibility would surface.
    """
    from openconcept.mission import FullMissionWithReserve

    class _MinimalAircraftModel(om.Group):
        """Smallest group satisfying OpenConcept's aircraft-model contract.

        This is not a test double for cdadt physics -- it exists only to prove
        OpenConcept's own machinery builds. cdadt's real adapter is exercised by the
        contract tests in ``cdadt/tests/mission/``.
        """

        def initialize(self):
            self.options.declare("num_nodes", default=1, types=int)
            self.options.declare("flight_phase", default=None, types=str)

        def setup(self):
            nn = self.options["num_nodes"]
            ivc = self.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
            ivc.add_output("thrust", val=np.full(nn, 1.0e5), units="N")
            ivc.add_output("drag", val=np.full(nn, 5.0e4), units="N")
            ivc.add_output("weight", val=np.full(nn, 5.0e4), units="kg")

    prob = om.Problem()
    prob.model.add_subsystem(
        "mission",
        FullMissionWithReserve(num_nodes=3, aircraft_model=_MinimalAircraftModel),
        promotes_inputs=["ac|*"],
    )
    prob.setup(check=False)

    # The variable that closes the sizing loop must exist after setup.
    assert prob.model.mission.loiter is not None


def test_openconcept_is_the_local_editable_clone():
    """OpenConcept resolves to a git working tree, not a wheel from PyPI.

    cdadt installs OpenConcept editable from the sibling clone so that local
    compatibility patches remain in effect. If this fails, the install step in
    ``docs/install.rst`` was skipped or a PyPI wheel shadowed the clone.
    """
    repo_root = openconcept_repo_root()
    assert repo_root is not None, (
        f"openconcept imported from {openconcept_package_dir()}, which is not inside a git "
        "working tree. Install it with: pip install -e <path to openconcept clone> --no-deps"
    )


def test_ipopt_is_available_and_solves_a_constrained_problem():
    """IPOPT is present through pyOptSparse and drives a real constrained problem to optimum.

    The problem is minimize ``x**2 + y**2`` subject to ``x + y >= 2``, whose optimum is
    ``x = y = 1``. Asserting the answer -- not just that a driver object exists -- is what
    distinguishes a working IPOPT build from an importable one.
    """
    if "IPOPT" not in available_pyoptsparse_optimizers():
        pytest.fail(
            "IPOPT is not available through pyOptSparse in this environment. cdadt_env must be "
            "created with: conda create -n cdadt_env -c conda-forge ... pyoptsparse ipopt cyipopt"
        )

    prob = om.Problem()
    prob.model.add_subsystem(
        "quad",
        om.ExecComp(["f = x**2 + y**2", "g = x + y"]),
        promotes=["*"],
    )
    prob.model.add_design_var("x", lower=-10.0, upper=10.0)
    prob.model.add_design_var("y", lower=-10.0, upper=10.0)
    prob.model.add_objective("f")
    prob.model.add_constraint("g", lower=2.0)

    prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    prob.driver.opt_settings["print_level"] = 0
    prob.driver.options["print_results"] = False

    prob.setup(check=False)
    prob.set_val("x", 5.0)
    prob.set_val("y", -3.0)
    prob.run_driver()

    assert prob.get_val("x").item() == pytest.approx(1.0, abs=1e-5)
    assert prob.get_val("y").item() == pytest.approx(1.0, abs=1e-5)
    assert prob.get_val("f").item() == pytest.approx(2.0, abs=1e-5)


def test_numpy_and_scipy_available():
    """NumPy and SciPy import, and NumPy provides the complex-step support cdadt relies on."""
    import scipy

    assert scipy.__version__
    # Complex step is how every cdadt-authored partial is verified; confirm it works here.
    assert np.imag(np.sqrt(np.complex128(4.0 + 1e-30j))) == pytest.approx(2.5e-31, rel=1e-6)
