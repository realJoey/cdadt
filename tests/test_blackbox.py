"""Unit and integration: loading, addressing and reading the black box."""

from __future__ import annotations

import pytest

from cdadt import BlackBoxError, OpenConceptSizingBox, SolverSettings

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
