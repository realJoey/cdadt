"""Unit and integration: loading, addressing and reading the black box."""

from __future__ import annotations

from pathlib import Path

import pytest

from cdadt import BlackBoxError, OpenConceptSizingBox, SolverSettings
from cdadt.blackbox import RunDirectory

MODEL = "openconcept.examples.B738_sizing:B738SizingMissionAnalysis"


# =============================================================================================
# Loading by name
# =============================================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openconcept.examples.B738_sizing.B738SizingMissionAnalysis", "must be given as"),
        ("no.such.module:Thing", "Cannot import the module"),
        ("openconcept.examples.B738_sizing:NoSuchClass", "has no attribute"),
        ("openconcept.examples.B738_sizing:run_738_sizing_analysis", "not a class"),
    ],
)
def test_a_bad_model_name_says_what_was_tried(spec, expected):
    """A mistyped model in a case file is otherwise an import traceback with no context."""
    with pytest.raises(BlackBoxError, match=expected):
        OpenConceptSizingBox(spec, num_nodes=5)


@pytest.mark.unit
def test_an_even_grid_is_refused():
    """The box integrates fuel burn with Simpson's rule, which needs 2N + 1 points."""
    with pytest.raises(ValueError, match="must be odd"):
        OpenConceptSizingBox(MODEL, num_nodes=10)


@pytest.mark.unit
def test_the_box_must_be_built_before_it_is_used():
    """The message names the method to call rather than raising an attribute error."""
    box = OpenConceptSizingBox(MODEL, num_nodes=5)
    assert not box.is_built
    with pytest.raises(BlackBoxError, match="has not been built"):
        _ = box.problem


# =============================================================================================
# The interface
# =============================================================================================


@pytest.mark.integration
def test_describe_reads_the_interface_without_running_anything():
    """Cheap enough to run before a long study, which is the point of it."""
    box = OpenConceptSizingBox.describe(MODEL)
    settable = box.settable()

    assert "ac|geom|wing|S_ref" in settable
    assert "mission.mission_range" in settable
    # Computed quantities are readable but never settable, which is what makes them unusable
    # as design variables and what the check exists to catch.
    assert "ac|weights|OEW" not in settable
    assert "ac|geom|hstab|S_ref" not in settable
    assert "ac|weights|OEW" in box.readable()


@pytest.mark.integration
def test_the_interface_does_not_depend_on_the_grid():
    """Which is why a three-node probe can validate a twenty-one-node study."""
    coarse = OpenConceptSizingBox.describe(MODEL, num_nodes=3)
    finer = OpenConceptSizingBox.describe(MODEL, num_nodes=5)
    assert set(coarse.settable()) == set(finer.settable())
    assert set(coarse.readable()) == set(finer.readable())


@pytest.mark.integration
def test_setting_a_computed_output_is_reported_with_suggestions():
    """A design variable that is not an independent variable fails here, not inside setup."""
    box = OpenConceptSizingBox.describe(MODEL)
    with pytest.raises(BlackBoxError, match="did you mean"):
        box.check_settable(["ac|geom|wing|Sref"])
    # The looser check exists for solver starting guesses, which are outputs on purpose.
    box.check_addressable(["ac|weights|MTOW"])
    with pytest.raises(BlackBoxError, match="not variables of"):
        box.check_addressable(["ac|weights|MTOW_guess"])


@pytest.mark.integration
def test_reading_something_the_box_does_not_publish_names_the_box():
    """So the reader knows which model was asked, not only that a key was missing."""
    box = OpenConceptSizingBox.describe(MODEL)
    with pytest.raises(BlackBoxError, match="does not publish"):
        box.get("ac|geom|wing|thickness_distribution")


@pytest.mark.integration
def test_scalars_come_back_as_numbers_and_vectors_as_arrays(built_box):
    """A one-element array propagates into every report and comparison downstream."""
    assert isinstance(built_box.get("ac|weights|MTOW", units="kg"), float)
    assert built_box.get("mission.climb.throttle").shape == (built_box.num_nodes,)


@pytest.mark.integration
def test_units_are_converted_on_the_way_out(built_box):
    """Reading in the wrong units is the quietest way to get a wrong number."""
    kilograms = built_box.get("ac|weights|MTOW", units="kg")
    pounds = built_box.get("ac|weights|MTOW", units="lb")
    assert pounds == pytest.approx(kilograms * 2.2046226218, rel=1e-6)


# =============================================================================================
# Solver configuration
# =============================================================================================


@pytest.mark.unit
def test_solver_settings_default_to_the_reference_run_scripts_own():
    """cdadt does not invent convergence criteria for somebody else's model."""
    settings = SolverSettings()
    assert (settings.maxiter, settings.atol, settings.rtol) == (20, 1e-9, 1e-9)
    assert settings.err_on_non_converge is True


# =============================================================================================
# Representations, accessors and error paths
# =============================================================================================


@pytest.mark.unit
def test_solver_settings_and_variable_info_repr_as_what_they_hold():
    """Both appear in debugger frames and in the interface listing."""
    from cdadt.blackbox import VariableInfo

    assert repr(SolverSettings(maxiter=25, atol=1e-8, rtol=1e-8)) == (
        "SolverSettings(maxiter=25, atol=1e-08, rtol=1e-08)"
    )
    info = VariableInfo("ac|geom|wing|S_ref", "m**2", (1,), "output")
    assert info.name == "ac|geom|wing|S_ref"
    assert info.kind == "output"
    assert info.is_scalar
    assert not VariableInfo("mission.climb.throttle", None, (11,), "output").is_scalar
    assert repr(info) == "VariableInfo('ac|geom|wing|S_ref', units='m**2', shape=(1,))"


@pytest.mark.unit
def test_an_unbuilt_box_reprs_as_unbuilt_and_publishes_nothing():
    """``has`` must answer, not raise, before the model exists -- it is a question about it."""
    box = OpenConceptSizingBox(MODEL, num_nodes=5)
    assert box.model_class.__name__ == "B738SizingMissionAnalysis"
    assert box.solver.maxiter == 20
    assert not box.has("ac|weights|MTOW")
    assert repr(box) == f"OpenConceptSizingBox({MODEL!r}, num_nodes=5, not built)"


@pytest.mark.integration
def test_a_built_box_reprs_as_built_and_exposes_its_model():
    """The group instance is what design variables are declared on before setup."""
    box = OpenConceptSizingBox.describe(MODEL)
    assert repr(box) == f"OpenConceptSizingBox({MODEL!r}, num_nodes=3, built)"
    assert box.model is box.problem.model
    assert box.model.__class__.__name__ == "B738SizingMissionAnalysis"


@pytest.mark.integration
def test_setting_a_variable_the_box_does_not_have_names_the_box():
    """So the reader knows which model refused it, not merely that a key was missing."""
    box = OpenConceptSizingBox.describe(MODEL)
    with pytest.raises(BlackBoxError, match="Cannot set"):
        box.set("ac|geom|wing|thickness_distribution", 0.12)


@pytest.mark.integration
def test_a_driver_can_be_attached_when_the_box_is_built():
    """The optimizer attaches one after build; this is the path that takes it at build time."""
    import openmdao.api as om

    box = OpenConceptSizingBox(MODEL, num_nodes=3, solver=SolverSettings(maxiter=0, err_on_non_converge=False))
    driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=1)
    box.build(driver=driver)
    assert box.problem.driver is driver


@pytest.mark.integration
def test_the_shape_of_a_variable_the_box_does_not_publish_is_read_from_the_model():
    """Neither settable nor readable, but still addressable -- so the shape comes from the value.

    ``shape_of`` exists to resample an initial condition written as two endpoints onto the grid
    the box declares, and it answers from ``readable``/``settable`` when it can. Not every
    addressable name is in either: a promoted input is omitted from both, because the name can
    reach several components at once. The ground-roll phases carry such names -- OpenConcept
    promotes the takeoff phases' integrator outputs as inputs of the balanced-field group -- and
    without this fallback a mission writing to one would fail on a name that plainly exists.
    """
    box = OpenConceptSizingBox(MODEL, num_nodes=3, solver=SolverSettings(maxiter=0, err_on_non_converge=False))
    box.build()

    name = "mission.v0v1.ode_integ_phase.range_final"
    assert name not in box.readable(), "no longer an unpublished name, so this test is moot"
    assert name not in box.settable(), "no longer an unpublished name, so this test is moot"
    assert box.has(name), "the fallback is only reached for a name that is addressable"

    assert box.shape_of(name) == (1,)
    # And the published path still answers from the declared shape rather than the fallback.
    assert box.shape_of("mission.climb.fltcond|h") == (3,)


# =============================================================================================
# Where a run writes
# =============================================================================================


@pytest.mark.unit
def test_a_run_directory_must_be_named():
    """An unnamed run has nowhere to put its files, and OpenMDAO would invent a name."""
    with pytest.raises(ValueError, match="needs a name"):
        RunDirectory("   ")


@pytest.mark.unit
def test_a_run_is_named_for_its_case_and_when_it_ran():
    """The stamp is passed in rather than read from a clock, so a run is reproducible by name."""
    run = RunDirectory.for_case("cases/b738.yaml", stamp="20260728_193000")

    assert run.name == "b738_20260728_193000"
    assert run.root == Path(RunDirectory.DEFAULT_ROOT)
    assert repr(run) == "RunDirectory('b738_20260728_193000', root='run_outputs')"


@pytest.mark.unit
def test_a_run_has_no_directory_until_its_problem_is_built():
    """OpenMDAO decides the path, so asking before it has been asked is an error, not a guess."""
    run = RunDirectory("unbuilt")
    with pytest.raises(BlackBoxError, match="has no directory yet"):
        _ = run.path


@pytest.mark.integration
def test_a_box_with_a_run_directory_writes_into_openmdaos_own_output_directory(tmp_path):
    """The claim the whole design rests on: cdadt's files and OpenMDAO's land in one place.

    ``.openmdao_out`` is OpenMDAO's own marker, and ``reports`` is written by OpenMDAO rather
    than by cdadt. Their presence is what says this directory is the problem's, not a folder
    made beside it -- which is why the driver's log ends up here too.
    """
    run = RunDirectory("probe_run", root=tmp_path)
    box = OpenConceptSizingBox(MODEL, num_nodes=3, solver=SolverSettings(maxiter=0, err_on_non_converge=False), run=run)
    box.build()

    assert box.run_directory is run
    assert run.path == tmp_path / "probe_run_out"
    assert (run.path / ".openmdao_out").exists()
    assert (run.path / "reports").is_dir()


@pytest.mark.integration
def test_building_a_box_without_a_run_directory_writes_nothing(tmp_path, monkeypatch):
    """An interface query must not scatter output folders through the working tree.

    Scoped to *building* on purpose, and the name says so. It would be wrong to read this as
    "a box without a run directory never writes": a driver writes its own log, and OpenMDAO
    creates the problem's output directory on demand to hold it, which ``reports=False`` does
    not prevent. That is why the command line always supplies a run directory, and why the test
    suite runs from a temporary working directory -- see
    :func:`tests.conftest.isolated_working_directory`.
    """
    monkeypatch.chdir(tmp_path)
    box = OpenConceptSizingBox(MODEL, num_nodes=3, solver=SolverSettings(maxiter=0, err_on_non_converge=False))
    box.build()

    assert box.run_directory is None
    assert not list(tmp_path.iterdir()), f"building scattered {[p.name for p in tmp_path.iterdir()]}"


@pytest.mark.unit
def test_the_box_reports_the_options_the_case_file_gave_it():
    """A run record has to say how the analysis was configured, not only which one it was.

    Empty for OpenConcept's own group, which declares nothing but the grid. cdadt's own analysis
    group takes its aerodynamics this way, so this is what distinguishes two runs of the same
    black box that computed different physics.
    """
    plain = OpenConceptSizingBox(MODEL, num_nodes=3)
    assert plain.model_options == {}

    configured = OpenConceptSizingBox(MODEL, num_nodes=3, options={"aerodynamic_loads": "some.module:Model"})
    assert configured.model_options == {"aerodynamic_loads": "some.module:Model"}

    # A copy, so a caller cannot reconfigure a built box by reaching through the accessor.
    configured.model_options["aerodynamic_loads"] = "something.else:Model"
    assert configured.model_options == {"aerodynamic_loads": "some.module:Model"}
