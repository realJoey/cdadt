"""Integration: cdadt's own aerodynamics installed in OpenConcept's mission.

The claims here are the ones the substitution rests on. The analysis group reproduces the
reference exactly with the parity model, so the wiring is proven; the loads model is genuinely
consulted, so the swap is not decorative; and the vortex lattice produces a physically coherent
aeroplane, checked against what theory says rather than against a stored number.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import openmdao.api as om
import pytest

from cdadt import Config, SizingAnalysis
from cdadt.adapter.aircraft import CdadtAircraftModel
from cdadt.adapter.analysis import SizingMissionAnalysis, jet_transport_zero_lift_drag
from cdadt.adapter.avl import MACH_SAMPLES, OpenAVLLoads, openavl_is_available
from cdadt.adapter.lattice import ANGLES_OF_ATTACK, DifferentiableLattice, LatticeLibrary, LatticePolar
from cdadt.adapter.loads import AerodynamicLoadsComp
from cdadt.adapter.oas import (
    ANGLES_OF_ATTACK_DEG,
    OpenAeroStructLattice,
    OpenAeroStructLoads,
    openaerostruct_is_available,
)
from cdadt.adapter.sections import WingSectionsComp
from cdadt.models import AeroCoefficients, AerodynamicLoads, FlightCondition, LoadsError, Planform, PolarLoads
from cdadt.models.planform import TrapezoidalPlanform
from tests.conftest import case_dict

#: Skip marker for everything that needs the optional vortex-lattice extra.
needs_openavl = pytest.mark.skipif(not openavl_is_available(), reason="openavl is not installed")

#: Skip marker for everything that needs OpenAeroStruct -- which cdadt reaches only through
#: OpenConcept, so what is probed is OpenConcept's subpackage rather than the package itself.
#:
#: Defined here beside its sibling rather than beside the tests that use it, which is where it was
#: first written. A decorator is evaluated when the module executes, top to bottom, so a marker
#: defined below its first use is a NameError at collection -- the whole file fails to import and
#: every test in it disappears rather than running.
needs_openaerostruct = pytest.mark.skipif(not openaerostruct_is_available(), reason="OpenAeroStruct is not installed")

#: The quantities the parity anchor compares, and the units to compare them in.
PARITY = {
    "ac|weights|MTOW": "kg",
    "ac|weights|OEW": "kg",
    "mission.descent.fuel_burn_integ.fuel_burn_final": "kg",
    "mission.loiter.fuel_burn_integ.fuel_burn_final": "kg",
    "mission.bfl.distance_continue": "ft",
    "mission.bfl.distance_abort": "ft",
    "mission.takeoff|v1": "kn",
    "ac|geom|hstab|S_ref": "m**2",
    "ac|geom|vstab|S_ref": "m**2",
    "ac|aero|CLmax_TO": None,
}


def _openavl_reference_geometry(name: str) -> Path | None:
    """Return the path to one of openavl's own reference geometries, or None if it is not there.

    Located from the installed package rather than assumed, the same way
    :mod:`tests.test_boundary` locates the OpenConcept clone -- and by walking up rather than by
    counting directories, because openavl installs from a ``src`` layout, so its package directory
    is one level deeper than OpenConcept's and a fixed number of ``.parent`` calls would be a guess
    that happens to work.

    The geometries are *read* and nothing else. The rule is that neither dependency is modified,
    moved or copied; validating cdadt against a dependency's own fixtures is the opposite of
    duplicating them.
    """
    import openavl

    for directory in Path(openavl.__file__).resolve().parents:
        candidate = directory / "tests" / "data" / "avl" / "geometries" / name
        if candidate.is_file():
            return candidate
    return None


def _cdadt_case(loads: str = "cdadt.models.polar:PolarLoads", nodes: int | None = None) -> Config:
    """Return the shipped sizing case, driven through cdadt's own analysis group."""
    data = copy.deepcopy(case_dict("b738.yaml"))
    data["black_box"]["model"] = "cdadt.adapter.analysis:SizingMissionAnalysis"
    data["black_box"]["options"] = {"aerodynamic_loads": loads}
    if nodes is not None:
        data["black_box"]["num_nodes"] = nodes
    return Config.from_dict(data)


# =============================================================================================
# The loads component
# =============================================================================================


@pytest.mark.unit
def test_the_component_evaluates_the_model_it_was_given():
    """The whole point: the drag published is the drag the model computed."""
    nodes = 3
    problem = om.Problem(reports=False)
    problem.model.add_subsystem(
        "loads", AerodynamicLoadsComp(num_nodes=nodes, loads_factory=PolarLoads), promotes=["*"]
    )
    problem.setup(force_alloc_complex=True)

    lift = np.array([0.4, 0.5, 0.6])
    problem.set_val("fltcond|CL", lift)
    problem.set_val("fltcond|q", np.full(nodes, 1.2e4), units="N/m**2")
    problem.set_val("fltcond|M", np.full(nodes, 0.78))
    problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    problem.set_val("ac|geom|wing|AR", 9.45)
    problem.set_val("ac|aero|polar|e", 0.82)
    problem.set_val("CD0", np.full(nodes, 0.019))
    problem.run_model()

    expected = (0.019 + lift**2 / (np.pi * 0.82 * 9.45)) * 1.2e4 * 124.6
    assert problem.get_val("drag", units="N") == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_the_components_partials_are_analytic_and_correct():
    """An optimizer steps on these, so they are checked rather than trusted.

    Checked against **finite differences**, not complex step, and the reason is a real limitation
    worth knowing: :class:`~cdadt.adapter.loads.AerodynamicLoadsComp` casts its inputs to ``float``
    to build the :class:`~cdadt.models.loads.FlightCondition` and the planform, which discards the
    imaginary part a complex step depends on. So the achievable agreement is bounded by finite
    difference truncation, and 1e-5 relative is what that allows -- the measured worst case is
    2.2e-6. Making the value objects complex-safe would let this be tightened to machine
    precision; it is not free, because every ``float()`` in that path would have to go.
    """
    nodes = 3
    problem = om.Problem(reports=False)
    problem.model.add_subsystem(
        "loads", AerodynamicLoadsComp(num_nodes=nodes, loads_factory=PolarLoads), promotes=["*"]
    )
    problem.setup(force_alloc_complex=True)
    problem.set_val("fltcond|CL", np.array([0.4, 0.5, 0.6]))
    problem.set_val("fltcond|q", np.full(nodes, 1.2e4), units="N/m**2")
    problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    problem.set_val("ac|geom|wing|AR", 9.45)
    problem.set_val("ac|aero|polar|e", 0.82)
    problem.set_val("CD0", np.full(nodes, 0.019))
    problem.run_model()

    checked = problem.check_partials(method="fd", out_stream=None)

    # The jacobians are compared directly rather than through OpenMDAO's summary "rel error".
    # That field reports infinity for ``ddrag/dCD0`` even though the two jacobians are identical
    # to five figures -- an artefact of how it is computed for a sparse-declared partial, not a
    # wrong derivative. Comparing the matrices says what is actually meant.
    compared = 0
    for (of, wrt), error in checked["loads"].items():
        analytic = np.asarray(error["J_fwd"])
        differenced = np.asarray(error["J_fd"])
        scale = max(np.abs(differenced).max(), 1.0)

        if np.abs(analytic).max() == 0.0 and np.abs(differenced).max() / scale < 1e-8:
            continue  # correctly zero, and not declared
        assert np.abs(differenced).max() > 0.0, f"d{of}/d{wrt} is declared but differences to nothing"
        assert np.abs(analytic - differenced).max() / scale < 1e-5, (
            f"d{of}/d{wrt} disagrees with finite differences by "
            f"{np.abs(analytic - differenced).max() / scale:.2e} relative"
        )
        compared += 1

    assert compared >= 5, f"only {compared} partials were actually compared; the check is vacuous"


# =============================================================================================
# The aircraft model
# =============================================================================================


@pytest.mark.unit
def test_a_model_that_needs_no_parasite_buildup_gets_a_zero_one():
    """A loads model may produce the whole drag coefficient itself.

    The builder is optional so that a future model owning the parasite drag as well as the
    induced drag needs no change here: it simply gets ``CD0`` of zero and adds its own.
    """
    problem = om.Problem(reports=False)
    problem.model.add_subsystem(
        "aircraft",
        CdadtAircraftModel(num_nodes=3, flight_phase="cruise", loads_factory=PolarLoads),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    problem.set_val("ac|geom|wing|AR", 9.45)
    problem.set_val("ac|aero|polar|e", 0.82)
    problem.set_val("fltcond|CL", np.full(3, 0.5))
    problem.set_val("fltcond|q", np.full(3, 1.2e4), units="N/m**2")
    problem.run_model()

    # With no parasite buildup, CD0 is zero and the drag is purely induced -- which is a
    # checkable number, not merely a non-None object.
    expected = (0.5**2 / (np.pi * 0.82 * 9.45)) * 1.2e4 * 124.6
    assert problem.get_val("zero_lift_drag.CD0").tolist() == [0.0, 0.0, 0.0]
    assert problem.get_val("drag", units="N") == pytest.approx(np.full(3, expected), rel=1e-10)


@pytest.mark.contract
def test_cdadts_aircraft_model_presents_the_same_face_as_openconcepts():
    """The mission consumes an aircraft model by name, so the two must publish the same names.

    ``CdadtAircraftModel`` stands where ``B738AircraftModel`` stands, and the mission reaches into
    it by name. Nothing checked that contract statically: a missing output would surface as an
    OpenMDAO connection error deep in a mission build, and a *spurious extra* one would not surface
    at all.

    What the two agree on is exactly ``drag``, ``thrust`` and ``weight`` -- measured from
    OpenConcept's real class rather than from a list written down here, because a written list is a
    copy of the dependency and would keep agreeing after the dependency changed. Note that fuel flow
    is **not** among them: the engine deck's fuel flow is connected to the fuel integrator inside the
    aircraft model, so it never crosses this boundary, and assuming it did was wrong.
    """
    from openconcept.examples.B738_sizing import B738AircraftModel

    def published(model_class: type, **extra: object) -> set[str]:
        """The variables a phase's aircraft model promotes to where the mission can see them."""
        problem = om.Problem(reports=False)
        problem.model.add_subsystem("acmodel", model_class(num_nodes=3, flight_phase="cruise", **extra), promotes=["*"])
        problem.setup(check=False)
        problem.final_setup()
        return {
            meta["prom_name"]
            for _absolute, meta in problem.model.list_outputs(prom_name=True, val=False, out_stream=None)
            if "." not in meta["prom_name"]  # promoted to the top is what the mission can reach
        }

    theirs = published(B738AircraftModel)
    ours = published(CdadtAircraftModel, loads_factory=PolarLoads)

    assert theirs == {"drag", "thrust", "weight"}, f"OpenConcept's aircraft model now publishes {sorted(theirs)}"
    assert ours == theirs, (
        f"the two aircraft models no longer present the same face to the mission.\n"
        f"  only OpenConcept's: {sorted(theirs - ours)}\n"
        f"  only cdadt's:       {sorted(ours - theirs)}"
    )


@pytest.mark.unit
def test_the_wing_sections_are_the_planforms_own_and_the_derivatives_are_analytic():
    """The sections OpenConcept's wave drag reads must be the wing cdadt's planform describes.

    Two things can go wrong and neither would raise. The consumer wants sections *"starting with the
    outboard section (wing tip) at the MOST NEGATIVE y value and moving inboard"*, so a reversed
    order would silently describe a wing with the tip chord at the root; and ``y_sec`` excludes the
    root, whose station is zero by that convention, so publishing both would shift the wing.
    """
    problem = om.Problem(reports=False)
    problem.model.add_subsystem("sections", WingSectionsComp(), promotes=["*"])
    problem.setup(force_alloc_complex=True)
    problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    problem.set_val("ac|geom|wing|AR", 9.45)
    problem.set_val("ac|geom|wing|taper", 0.159)
    problem.set_val("ac|geom|wing|toverc", 0.1)
    problem.run_model()

    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    root, tip = wing.sections()
    assert problem.get_val("y_sec", units="m")[0] == pytest.approx(-tip.y, rel=1e-12), "the tip station is not negative"
    assert problem.get_val("chord_sec", units="m") == pytest.approx([tip.chord, root.chord], rel=1e-12)
    assert problem.get_val("toverc_sec") == pytest.approx([0.1, 0.1])

    checked = problem.check_partials(method="fd", out_stream=None)
    worst = 0.0
    for (of, wrt), error in checked["sections"].items():
        analytic, differenced = np.asarray(error["J_fwd"]), np.asarray(error["J_fd"])
        scale = max(np.abs(differenced).max(), 1e-12)
        assert np.abs(differenced).max() > 0.0, f"d{of}/d{wrt} is declared but differences to nothing"
        worst = max(worst, np.abs(analytic - differenced).max() / scale)
    assert worst < 1e-5, f"the section derivatives disagree with finite differences by {worst:.2e}"


@needs_openaerostruct
@pytest.mark.integration
def test_openconcepts_own_wave_drag_can_be_installed_and_is_off_by_default():
    """The transonic gap was real, and it is closed with the dependency's own component.

    Neither the vortex lattice nor OpenConcept's jet-transport parasite buildup carries a Mach term,
    so the shipped aeroplane cruised at M 0.785 with no drag rise anywhere in the model. But
    OpenConcept ships ``WaveDragFromSections`` -- a Korn-equation model verified against
    OpenAeroStruct by its own ``test_wave_drag.py`` -- so the fix was to install it, not to write one.

    **Off by default, and that default is load-bearing.** ``B738AircraftModel`` has no wave drag, so
    an aircraft model that added it unasked could not reproduce the reference example, and the parity
    anchor every other comparison rests on would be gone.
    """
    nodes = 3

    def built(wave_drag: bool):
        """A cruise aircraft model with and without the transonic term."""
        problem = om.Problem(reports=False)
        problem.model.add_subsystem(
            "aircraft",
            CdadtAircraftModel(
                num_nodes=nodes,
                flight_phase="cruise",
                loads_factory=PolarLoads,
                zero_lift_drag_builder=jet_transport_zero_lift_drag,
                wave_drag=wave_drag,
            ),
            promotes=["*"],
        )
        problem.setup(check=False)
        problem.final_setup()
        return problem

    def subsystems(problem) -> set[str]:
        """The aircraft model's immediate children, by name, through OpenMDAO's public iterator."""
        return {system.name for system in problem.model.aircraft.system_iter(recurse=False)}

    assert "wave_drag" not in subsystems(built(False)), "wave drag is on by default"
    assert "wave_drag" in subsystems(built(True)), "asking for wave drag did not add it"

    problem = built(True)
    for name, value, units in (
        ("ac|geom|wing|S_ref", 124.6, "m**2"),
        ("ac|geom|wing|AR", 9.45, None),
        ("ac|geom|wing|taper", 0.159, None),
        ("ac|geom|wing|c4sweep", 25.0, "deg"),
        ("ac|geom|wing|toverc", 0.1, None),
        ("ac|aero|polar|e", 0.801, None),
    ):
        problem.set_val(name, value, units=units)
    problem.set_val("fltcond|CL", np.full(nodes, 0.5))
    problem.set_val("fltcond|M", np.array([0.5, 0.785, 0.82]))
    problem.run_model()

    wave = problem.get_val("wave_drag.CD_wave")
    # Zero well below the drag-rise Mach, and growing steeply through it. That shape is the model:
    # a term that is flat and then not is what a Korn-equation drag rise looks like.
    assert wave[0] == pytest.approx(0.0, abs=1e-12), "wave drag appears at M 0.5, below drag rise"
    assert 0.0 < wave[1] < wave[2], f"wave drag does not grow through the drag rise: {wave}"
    assert wave[2] > 10.0 * wave[1], "the drag rise is not steep enough to be a drag rise"

    # And it reaches the drag the mission consumes, rather than being computed and dropped.
    total = problem.get_val("total_zero_lift_drag.CD0")
    parasite = problem.get_val("zero_lift_drag.CD0")
    assert total == pytest.approx(parasite + wave, rel=1e-12)


@pytest.mark.unit
def test_the_takeoff_phases_get_the_takeoff_drag_configuration():
    """The flaps are down on the ground, and the parasite buildup is configured for it."""
    assert CdadtAircraftModel.TAKEOFF_PHASES == ("v0v1", "v1v0", "v1vr", "rotate")
    takeoff = jet_transport_zero_lift_drag(3, in_takeoff=True)
    clean = jet_transport_zero_lift_drag(3, in_takeoff=False)
    assert takeoff.options["configuration"] == "takeoff"
    assert clean.options["configuration"] == "clean"


# =============================================================================================
# The analysis group: the parity anchor
# =============================================================================================


@pytest.mark.validation
@pytest.mark.slow
def test_cdadt_own_analysis_group_reproduces_the_reference(reference_problem):
    """The anchor. cdadt's group, with cdadt's drag model, must reproduce OpenConcept exactly.

    Everything about this path is cdadt's -- the analysis group, the aircraft model, the component
    that evaluates the loads, the polar itself. Only the equation is shared. Reproducing the
    reference to round-off is what makes any *later* difference attributable to new physics
    instead of to new plumbing, and it is the reason the parabolic polar exists at all.
    """
    analysis = SizingAnalysis(_cdadt_case())
    analysis.build()
    analysis.converge()

    mismatches = []
    for path, units in PARITY.items():
        theirs = reference_problem.get_val(path, units=units).item()
        ours = float(np.atleast_1d(np.asarray(analysis.box.get(path, units=units))).reshape(-1)[0])
        error = abs(ours - theirs) / abs(theirs) if theirs else abs(ours)
        if error >= 1e-9:
            mismatches.append(f"  {path}: cdadt {ours!r} vs OpenConcept {theirs!r} (rel {error:.3e})")

    assert not mismatches, "cdadt's own analysis group does not reproduce the reference:\n" + "\n".join(mismatches)


@pytest.mark.unit
def test_the_case_file_chooses_the_aerodynamics():
    """A study changes its aerodynamics by naming a different class, and nothing else."""
    assert _cdadt_case().black_box.options == {"aerodynamic_loads": "cdadt.models.polar:PolarLoads"}

    group = SizingMissionAnalysis(num_nodes=3, aerodynamic_loads="cdadt.models.polar:PolarLoads")
    assert group._loads_factory() is PolarLoads


@pytest.mark.unit
def test_naming_something_that_is_not_a_loads_model_is_refused_before_setup():
    """Caught where the case file can be blamed, not as an attribute error mid-solve."""
    with pytest.raises(LoadsError, match="not an AerodynamicLoads"):
        SizingMissionAnalysis(
            num_nodes=3, aerodynamic_loads="cdadt.models.planform:TrapezoidalPlanform"
        )._loads_factory()

    with pytest.raises(LoadsError, match="Cannot import the module"):
        SizingMissionAnalysis(num_nodes=3, aerodynamic_loads="no.such.module:Model")._loads_factory()


@pytest.mark.integration
def test_the_group_accepts_the_aeroplane_and_publishes_the_mission():
    """Built once cheaply: the interface has to be right before a convergence is worth trying."""
    problem = om.Problem(model=SizingMissionAnalysis(num_nodes=3), reports=False)
    problem.setup(check=False)
    problem.final_setup()

    outputs = problem.model.get_io_metadata("output", return_rel_names=False)
    assert "mission.climb.acmodel.aero_loads.drag" in outputs, "cdadt's loads are not in the mission"
    assert "takeoff_weight.MTOW" in outputs
    assert "empty_weight.sum_weights.OEW" in outputs

    settable = [
        name
        for name, meta in problem.model.get_io_metadata("output", metadata_keys=["tags"]).items()
        if "openmdao:indep_var" in meta["tags"]
    ]
    assert any(name.endswith("ac|geom|wing|S_ref") for name in settable)


# =============================================================================================
# The vortex lattice
# =============================================================================================


@needs_openavl
@pytest.mark.unit
def test_a_rectangular_wing_has_the_span_efficiency_theory_says_it_should():
    """The check that validates the lattice setup itself.

    For a planar wing the elliptical loading is optimal and gives exactly 1.0, so a rectangular
    wing must come out a little below. This is what confirmed the reference areas, the units and
    the section placement were right -- and it is what caught the model reading near-field drag,
    which put the swept wing above 1.0, where no planar wing can be.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=0.0, taper=1.0)
    efficiency = OpenAVLLoads(planform=wing).span_efficiency

    assert 0.9 < efficiency < 1.0, f"a planar rectangular wing cannot have e = {efficiency}"


@needs_openavl
@pytest.mark.unit
def test_the_shipped_wing_gets_a_physical_span_efficiency_from_the_lattice():
    """And it disagrees with the case file's assumption, which is the reason to compute it."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    efficiency = OpenAVLLoads(planform=wing).span_efficiency

    assert 0.0 < efficiency <= 1.0, "a planar wing cannot exceed elliptical efficiency"
    assert efficiency > 0.82, "the lattice should find this wing better than the assumed 0.82"


@needs_openavl
@pytest.mark.validation
def test_the_fitted_curvature_is_openavls_own_span_efficiency():
    """The identity that says the polar is paired with the right two coefficients.

    For a planar wing the far-field drag and lift satisfy ``CDFF = CLFF^2 / (pi e AR)`` with the
    ``e`` openavl reports in ``SPANEF``. So the fitted curvature must equal ``1 / (pi e AR)`` to
    round-off, and nothing else does: pairing the far-field drag with the *commanded* near-field
    lift -- which this model did -- biased the curvature by 2.2% and left the reported ``e`` of
    0.990 inconsistent with the 0.969 the drag implied. This test is what makes that class of
    mistake impossible to reintroduce quietly.
    """
    aspect = 9.45
    polar = LatticeLibrary().polar_for(
        TrapezoidalPlanform(area=124.6, aspect_ratio=aspect, sweep=25.0, taper=0.159), 0.0, 6, 20
    )
    implied = 1.0 / (np.pi * polar.span_efficiency * aspect)

    assert polar.curvature == pytest.approx(implied, rel=1e-12), (
        f"the fitted curvature {polar.curvature:.10f} implies e = "
        f"{1.0 / (np.pi * polar.curvature * aspect):.6f}, but openavl reports "
        f"{polar.span_efficiency:.6f}"
    )


@needs_openavl
@pytest.mark.validation
def test_the_lattice_polar_is_exactly_quadratic():
    """Three solves determine the polar *exactly*, and this measures that rather than assuming it.

    The claim rests on the lattice being linear: the circulation is affine in angle of attack, so
    the far-field lift is linear in it and the far-field drag a quadratic form. If that were only
    approximately true, the closed-form evaluation every mission node uses would be a surrogate
    with an error budget, and the error would have to be published instead of the identity.
    """
    lattice = DifferentiableLattice(
        TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159), 0.0, 6, 20
    )
    polar = lattice.fit()

    worst = 0.0
    for angle in (0.04, 0.12, 0.20):  # none of them sampled by the fit
        assert angle not in ANGLES_OF_ATTACK
        lift, drag, _efficiency = lattice.far_field_at(angle)
        worst = max(worst, abs(float(polar.drag_at(np.asarray(lift))) - drag) / drag)

    assert worst < 1e-12, f"the polar is not exactly quadratic: it drifts by {worst:.2e} off its samples"


@needs_openavl
@pytest.mark.validation
def test_cdadts_composition_is_openavls_own_composition():
    """cdadt assembles openavl's differentiable chain by hand. This proves it assembled it right.

    ``run_analysis_with_geometry`` is openavl's own composition of the four calls, and it is what
    openavl's ``OpenAVLComp`` -- the component its ``rectangular_wing_opt.py`` example drives --
    uses. cdadt cannot call it, because it returns an ``AnalysisResult``, which keeps only the
    near-field drag and discards the far-field pair this model is built on.

    So cdadt composes the same four public calls itself, and the risk that creates is a subtly
    different assembly: a reference quantity applied in the wrong place, a velocity pair swapped.
    On the quantities the two do share -- lift and near-field drag -- they must therefore agree to
    round-off, and the measured agreement is 3e-16 relative, one bit in the last place. Not exactly
    equal, because the reference quantities on this side are assembled in JAX from the design vector
    and on that side from Python floats, so the two arrive at the same numbers by slightly different
    arithmetic. Anything larger than round-off would mean a different assembly, not a different
    rounding.
    """
    from openavl.jax import ReferenceQuantities, run_analysis_with_geometry
    from openavl.jax.backend import jnp
    from openavl.jax.types import FlowCondition, GeometryDesignParams

    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    lattice = DifferentiableLattice(wing, 0.0, 6, 20)
    numbers = (wing.area, wing.aspect_ratio, wing.sweep, wing.taper)
    geometry = TrapezoidalPlanform.geometry(*numbers)

    theirs_arguments = (
        GeometryDesignParams(
            aincs=lattice._incidences,
            chords=jnp.asarray([geometry["root_chord"], geometry["tip_chord"]]),
            xles=jnp.asarray([0.0, geometry["tip_leading_edge"]]),
            yles=jnp.asarray([0.0, geometry["semi_span"]]),
            zles=jnp.zeros(2),
        ),
        lattice._topology,
        lattice._baseline,
        ReferenceQuantities(
            sref=jnp.asarray(wing.area),
            cref=jnp.asarray(geometry["mean_aerodynamic_chord"]),
            bref=jnp.asarray(geometry["span"]),
            xyzref=jnp.asarray([0.25 * geometry["mean_aerodynamic_chord"], 0.0, 0.0]),
            cdref=jnp.zeros(()),
            iysym=lattice._symmetry,
        ),
    )

    for angle in ANGLES_OF_ATTACK[1:]:  # alpha = 0 makes both sides trivially zero
        ours = lattice._far_field_forces(jnp.asarray(angle), jnp.asarray(numbers))
        theirs = run_analysis_with_geometry(
            FlowCondition(
                alfa=jnp.asarray(angle),
                beta=jnp.zeros(()),
                wrot=jnp.zeros(3),
                mach=jnp.zeros(()),
                delcon=jnp.zeros(lattice._controls),
            ),
            *theirs_arguments,
        )
        assert float(ours.CL) == pytest.approx(float(theirs.CL), rel=1e-14), f"lift differs at alpha {angle}"
        assert float(ours.CD) == pytest.approx(float(theirs.CD), rel=1e-14), f"drag differs at alpha {angle}"


@needs_openavl
@needs_openaerostruct
@pytest.mark.validation
@pytest.mark.slow
def test_openavl_and_openconcepts_own_vortex_lattice_agree_on_this_wing():
    """Two independent vortex lattices, one wrapped by each dependency, on the identical wing.

    This is the anchor for cdadt's aerodynamics that no amount of internal consistency can provide.
    OpenConcept's own alternative to a parabolic polar is OpenAeroStruct, reached through
    ``VLMDragPolar``; its underlying ``VLM`` reports ``fltcond|CDi`` separately from viscous and wave
    drag, and ``TrapezoidalPlanformMesh`` builds its mesh from the same four numbers cdadt's planform
    uses. So the same wing can be put through both codes.

    **Compared near-field to near-field**, which is the whole subtlety. OpenAeroStruct's ``CDi`` is a
    sum of panel forces, so it must be compared against openavl's ``CD`` and not against the
    Trefftz-plane ``CDFF`` cdadt actually flies on -- those differ by 5.3% *within openavl*, and
    comparing across that boundary would report a disagreement between quantities as a disagreement
    between codes.

    And the reason cdadt flies on the far field is confirmed here by the other dependency: both
    codes' near-field values imply a span efficiency **above 1** for this planar wing, which is
    impossible, since elliptical loading is optimal. OpenConcept's own VLM has the same limitation,
    independently, which is stronger evidence for the far-field choice than cdadt's own reasoning
    about it.
    """
    from openavl.jax.backend import jnp
    from openconcept.aerodynamics.openaerostruct import VLM, TrapezoidalPlanformMesh

    area, aspect, sweep, taper, thickness = 124.6, 9.45, 25.0, 0.159, 0.1
    target, panels_x, panels_y = 0.5, 6, 20

    problem = om.Problem(reports=False)
    problem.model.add_subsystem(
        "mesh", TrapezoidalPlanformMesh(num_x=panels_x, num_y=panels_y), promotes_outputs=[("mesh", "OAS_mesh")]
    )
    problem.model.add_subsystem(
        "vlm",
        # Induced drag only from both sides: openavl's lattice has no viscous or wave term, so
        # leaving OpenAeroStruct's on would compare two different quantities.
        VLM(num_x=panels_x, num_y=panels_y, surf_options={"with_viscous": False, "with_wave": False}),
        promotes_inputs=[("ac|geom|wing|OAS_mesh", "OAS_mesh")],
    )
    problem.setup()
    problem.set_val("mesh.S", area, units="m**2")
    problem.set_val("mesh.AR", aspect)
    problem.set_val("mesh.taper", taper)
    problem.set_val("mesh.sweep", sweep, units="deg")
    problem.set_val("vlm.ac|geom|wing|toverc", np.full(panels_y, thickness))
    # Not M = 0: OpenAeroStruct works in true airspeed and derives it from Mach and altitude, so
    # zero Mach is a zero velocity and every coefficient comes back NaN. Its lattice is
    # incompressible anyway -- CDi is identical at M 0.3 and 0.5 -- so the value is immaterial.
    problem.set_val("vlm.fltcond|M", 0.3)
    problem.set_val("vlm.fltcond|h", 0.0, units="m")

    def trimmed(evaluate, low: float, high: float) -> float:
        """Bisect to the target lift coefficient. Both lattices are linear, so this converges."""
        for _ in range(60):
            middle = 0.5 * (low + high)
            if evaluate(middle) < target:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def openaerostruct_lift(alpha_degrees: float) -> float:
        """OpenAeroStruct's lift coefficient at one angle of attack."""
        problem.set_val("vlm.fltcond|alpha", alpha_degrees, units="deg")
        problem.run_model()
        return float(problem.get_val("vlm.fltcond|CL")[0])

    openaerostruct_lift(trimmed(openaerostruct_lift, -5.0, 15.0))
    theirs = float(problem.get_val("vlm.fltcond|CDi")[0])

    lattice = DifferentiableLattice(
        TrapezoidalPlanform(area=area, aspect_ratio=aspect, sweep=sweep, taper=taper), 0.0, panels_x, panels_y
    )
    design = jnp.asarray([area, aspect, sweep, taper])
    solved = lattice._far_field_forces(
        jnp.asarray(trimmed(lambda a: float(lattice._far_field_forces(jnp.asarray(a), design).CL), -0.1, 0.3)), design
    )
    ours, ours_far_field = float(solved.CD), float(solved.CDFF)

    disagreement = abs(ours / theirs - 1.0)
    assert disagreement < 0.02, (
        f"openavl and OpenAeroStruct now disagree by {disagreement:.2%} on the induced drag of the "
        f"same wing ({ours:.8f} against {theirs:.8f}); it was 0.67%"
    )
    for name, near_field in (("OpenAeroStruct", theirs), ("openavl", ours)):
        implied = target**2 / (np.pi * aspect * near_field)
        assert implied > 1.0, (
            f"{name}'s near-field induced drag no longer implies the impossible e = {implied:.4f} "
            f"for this planar wing, which is the evidence cdadt reads the far field instead"
        )
    assert ours_far_field > ours, "the far-field drag must exceed the near-field value on a swept wing"


@needs_openavl
@pytest.mark.validation
def test_what_the_simplified_wing_costs_against_openavls_own_737_model():
    """openavl ships a detailed 737-800. cdadt models the same wing with two untwisted sections.

    This is the one check here that compares cdadt against reality rather than against its own
    consistency, and it does not pass by being close. ``b737.avl`` is openavl's own reference
    geometry -- the one its ``reference``-marked ``test_derivatives_three_way`` runs on -- with
    twist, dihedral, a kink, a real airfoil and control surfaces. cdadt's
    :class:`~cdadt.models.planform.TrapezoidalPlanform` has none of those.

    The difference is **reported, not tolerated**: the assertions bound it loosely enough to be a
    regression guard and the message carries the number, because the number is the finding. It
    matters because the lattice's span efficiency is what drives the whole fuel result, so an
    optimistic wing makes an optimistic aeroplane. See :doc:`/validation`.
    """
    from openavl import AVLSolver

    reference_geometry = _openavl_reference_geometry("b737.avl")
    if reference_geometry is None:
        pytest.skip("openavl's own reference 737 geometry is not present in the installed clone")

    solver = AVLSolver(str(reference_geometry))
    solver.set_parameter("cl", 0.5)
    solver.setup_trim(mode=1)
    solver.execute_run(max_iter=40)
    detailed = float(solver.get_results()["SPANEF"])
    detailed_aspect_ratio = float(solver.state.bref) ** 2 / float(solver.state.sref)

    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    simplified = OpenAVLLoads(planform=wing).span_efficiency

    optimism = simplified / detailed - 1.0
    assert 0.0 < detailed < 1.0, f"openavl's own 737 reports a non-physical e of {detailed}"
    assert optimism > 0.0, (
        f"the simplified wing is no longer the optimistic one: it reports e = {simplified:.4f} "
        f"against {detailed:.4f} for openavl's detailed model, which would invert the caveat the "
        f"documentation carries"
    )
    assert optimism < 0.15, (
        f"cdadt's two-section untwisted trapezoid now reports a span efficiency {optimism:.1%} "
        f"above openavl's own detailed 737 model ({simplified:.4f} against {detailed:.4f}, at "
        f"aspect ratios {wing.aspect_ratio:.2f} and {detailed_aspect_ratio:.2f}). That was 6.7% "
        f"when the docs were written; a larger gap means the simplification costs more than "
        f"documented"
    )


@needs_openavl
@pytest.mark.validation
def test_the_polar_agrees_with_openavls_own_solver():
    """Cross-check across two openavl paths: the numpy solver and the JAX geometry rebuild.

    The values this model publishes come from ``compute_forces`` over a geometry rebuilt inside
    JAX. openavl's ``AVLSolver`` reaches the same physics by a different route -- the one its
    reference tests validate against the Fortran binaries -- so trimming it to a lift coefficient
    and reading the far-field pair from *its* results is an independent check that the JAX rebuild
    describes the same wing.
    """
    from openavl import Aircraft, AVLSolver

    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    polar = OpenAVLLoads(planform=wing).polar_at(0.0)
    aircraft = DifferentiableLattice.aircraft(wing, 0.0, 6, 20, Aircraft)

    worst = 0.0
    for target in (0.35, 0.45, 0.65):
        solver = AVLSolver(aircraft, cd0=0.0, rho=1.225)
        solver.set_parameter("cl", target)
        solver.setup_trim(mode=1)
        solver.execute_run(max_iter=30)
        results = solver.get_results()
        # The solver's own far-field pair -- CDFF read against the CLFF it achieved, not the CL it
        # was commanded to hold. That distinction is the whole defect this pins.
        solved_lift, solved_drag = float(results["CLFF"]), float(results["CDFF"])
        worst = max(worst, abs(float(polar.drag_at(np.asarray(solved_lift))) - solved_drag) / solved_drag)

    assert worst < 1e-6, f"the two openavl paths disagree about this wing by {worst:.2e}"


@needs_openavl
@pytest.mark.validation
def test_the_geometry_derivatives_are_exact_not_approximate():
    """Every wing number's derivative, against central differences of the model as installed.

    This replaces an approximation and two absences. The aspect-ratio term used to come from
    ``k = 1/(pi e AR)`` with the span efficiency held fixed, which is wrong by 1.8%; sweep and
    taper had no derivative at all, so an optimizer was told they did not matter -- and the taper
    term is comparable in size to the aspect-ratio one.

    Differenced through :meth:`OpenAVLLoads.coefficients`, so what is checked is the derivative of
    the number the mission actually consumes, including the Mach interpolation, and not of some
    inner quantity.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    condition = FlightCondition(CL=0.5, mach=0.6)
    analytic = OpenAVLLoads(planform=wing).drag_gradients(condition, wing)

    def drag_coefficient(area: float, aspect: float, sweep: float, taper: float) -> float:
        """CD from the model as installed, at the same flight condition."""
        moved = TrapezoidalPlanform(area=area, aspect_ratio=aspect, sweep=sweep, taper=taper)
        return float(OpenAVLLoads(planform=moved).coefficients(condition, moved).CD[0])

    base = (124.6, 9.45, 25.0, 0.159)
    steps = {"area": 0.05, "AR": 0.005, "sweep": 0.02, "taper": 0.0005}
    worst = {}
    for index, (name, step) in enumerate(steps.items()):
        moved_up, moved_down = list(base), list(base)
        moved_up[index] += step
        moved_down[index] -= step
        differenced = (drag_coefficient(*moved_up) - drag_coefficient(*moved_down)) / (2.0 * step)
        got = float(analytic[name][0])
        worst[name] = abs(got - differenced) / max(abs(differenced), 1e-12)

    assert worst["AR"] < 1e-5, f"dCD/dAR is off by {worst['AR']:.2e}"
    assert worst["sweep"] < 1e-5, f"dCD/dsweep is off by {worst['sweep']:.2e}"
    assert worst["taper"] < 1e-4, f"dCD/dtaper is off by {worst['taper']:.2e}"
    # The drag *coefficient* is normalised by the area it is computed on, so it genuinely does not
    # move with area. Asserting that it is zero is a stronger statement than differencing it.
    assert abs(float(analytic["area"][0])) < 1e-12, "the drag coefficient should not depend on area"
    # And the derivative that does exist must be large enough for the agreement to mean something.
    assert abs(float(analytic["taper"][0])) > 1e-5, "the taper derivative is too small to have been checked"


@needs_openavl
@pytest.mark.unit
def test_the_lattice_builds_the_same_wing_in_jax_as_the_planform_does_in_numpy():
    """One statement of the geometry, evaluated in two array libraries, must give one wing.

    The lattice differentiates with respect to the four wing numbers, so its geometry has to be
    evaluated on JAX tracers; :class:`~cdadt.models.planform.TrapezoidalPlanform` casts to ``float``
    and cannot hold one. The formulas are therefore written once and the array library is injected.
    This is the test that keeps that honest: if the two paths ever diverge, the symptom would be a
    wrong *derivative* on a right-looking drag, which nothing else here would catch.
    """
    from openavl.jax.backend import jnp

    numbers = (124.6, 9.45, 25.0, 0.159)
    in_numpy = TrapezoidalPlanform.geometry(*numbers)
    in_jax = TrapezoidalPlanform.geometry(*(jnp.asarray(value) for value in numbers), arrays=jnp)

    assert set(in_jax) == set(in_numpy)
    for name, expected in in_numpy.items():
        assert float(in_jax[name]) == pytest.approx(float(expected), rel=1e-7), f"{name} differs between the two"

    # And the sections the lattice is actually built from agree with the same dictionary.
    root, tip = TrapezoidalPlanform(*numbers).sections()
    assert (root.chord, tip.chord) == pytest.approx((in_numpy["root_chord"], in_numpy["tip_chord"]))
    assert (tip.x, tip.y) == pytest.approx((in_numpy["tip_leading_edge"], in_numpy["semi_span"]))


@needs_openavl
@pytest.mark.unit
def test_the_solved_polars_live_in_an_object_whose_lifetime_somebody_owns():
    """The lattice must be solved once per wing per study -- without a module-level cache.

    Both halves matter. Solving per evaluation would be unaffordable: the model is rebuilt at every
    Newton iteration by design, and a lattice costs tens of seconds. Solving into a module-level
    cache would be affordable and wrong, which is what this used to do.

    So the store is an object, the caller owns it, and the caller is the analysis group -- one
    library per study, shared by all fourteen phases. Checked by counting solves, because that is
    the property that actually costs money.
    """
    library = LatticeLibrary()
    assert len(library) == 0
    assert repr(library) == "LatticeLibrary(DifferentiableLattice, 0 solved)"

    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    condition = FlightCondition(CL=0.5, mach=0.0)

    # Two models built the way the component builds them, per evaluation, sharing one library.
    for _ in range(2):
        model = OpenAVLLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace=library)
        model.coefficients(condition, wing)
    solved_once = len(library)
    assert solved_once == len(MACH_SAMPLES), "one wing should cost one solve per Mach sample, however many models"

    # A different wing is a different problem and must be solved; the same wing must not.
    other = TrapezoidalPlanform(area=100.0, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    OpenAVLLoads.build(planform=other, span_efficiency=0.8, zero_lift_drag=0.0, workspace=library).coefficients(
        condition, other
    )
    assert len(library) == 2 * solved_once

    library.clear()
    assert len(library) == 0


@pytest.mark.unit
def test_a_loads_model_is_only_handed_a_workspace_it_can_use():
    """``workspace`` is deliberately untyped in the interface, so each model checks its own.

    The alternative -- a single workspace type on the ABC -- would make the base class know what
    every future model needs to keep, which is exactly the coupling the slot exists to avoid. The
    cost is that a model must refuse what it cannot store, by name, rather than failing later on an
    attribute.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)

    # A model with nothing expensive to keep ignores it entirely, whatever it is given.
    assert PolarLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace="nonsense")

    if openavl_is_available():
        with pytest.raises(LoadsError, match="cannot store anything in"):
            OpenAVLLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace="nonsense")


@needs_openavl
@pytest.mark.unit
def test_the_lattice_is_solved_at_the_mach_number_of_each_node():
    """A climb node and a cruise node are different aerodynamic problems.

    This model evaluated every node at M = 0 at first, so the whole mission got the incompressible
    answer. Compressibility reaches the lattice through openavl's Prandtl-Glauert correction, and
    what is checked here is that it arrives *and interpolates* rather than snapping to a sample.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    loads = OpenAVLLoads(planform=wing)

    def drag_at(mach: float) -> float:
        """CD at CL 0.5 and one Mach number."""
        return float(loads.coefficients(FlightCondition(CL=0.5, mach=mach), wing).CD[0])

    below, above = drag_at(0.3), drag_at(0.5)
    assert below != pytest.approx(above, rel=1e-6), "Mach reaches the lattice not at all"
    assert above < drag_at(0.4) < below, "M = 0.4 is not interpolated between its neighbours"

    # Outside the sampled range np.interp holds the end value, deliberately: a vortex lattice is
    # not a model of anything above the top sample, so extrapolating would invent confidence.
    assert drag_at(MACH_SAMPLES[-1] + 0.2) == pytest.approx(drag_at(MACH_SAMPLES[-1]), rel=1e-12)


@needs_openavl
@pytest.mark.unit
def test_the_component_declares_the_partials_the_installed_model_actually_has():
    """A parabolic polar and a lattice have different sparsity, and both must be told the truth.

    An undeclared partial is an assertion that the derivative is zero. Declaring sweep and taper
    for a model that does not read them earns an OpenMDAO warning about zero derivatives; not
    declaring them for the lattice would hide its largest geometry term.
    """
    polar = AerodynamicLoadsComp(num_nodes=3, loads_factory=PolarLoads)
    lattice = AerodynamicLoadsComp(num_nodes=3, loads_factory=OpenAVLLoads)
    for component in (polar, lattice):
        om.Problem(reports=False).model.add_subsystem("loads", component)

    assert polar._declared_scalars() == ("ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|aero|polar|e")
    assert lattice._declared_scalars() == (
        "ac|geom|wing|S_ref",
        "ac|geom|wing|AR",
        "ac|geom|wing|c4sweep",
        "ac|geom|wing|taper",
    )


@needs_openavl
@pytest.mark.unit
def test_the_polar_refuses_a_derivative_it_never_took():
    """Asking for a wing number the lattice was not differentiated against is an error.

    Returning zero would be indistinguishable from a variable that genuinely does not matter,
    which is exactly the confusion that let sweep and taper go missing in the first place.
    """
    polar = LatticeLibrary().polar_for(
        TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159), 0.0, 6, 20
    )

    assert polar.coefficients == (polar.minimum_drag, polar.curvature, polar.lift_at_minimum_drag)
    with pytest.raises(KeyError, match="no derivative with respect to 'twist'"):
        polar.gradient("twist")


@needs_openavl
@pytest.mark.unit
def test_the_lattice_model_ignores_the_case_files_span_efficiency():
    """It computes its own, so the stated value must have no influence -- including on gradients."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    condition = FlightCondition(CL=0.5, mach=0.0)

    optimistic = OpenAVLLoads.build(planform=wing, span_efficiency=0.99, zero_lift_drag=0.0)
    pessimistic = OpenAVLLoads.build(planform=wing, span_efficiency=0.50, zero_lift_drag=0.0)

    assert pytest.approx(pessimistic.coefficients(condition, wing).CD) == optimistic.coefficients(condition, wing).CD
    assert optimistic.drag_gradients(condition, wing)["e"].tolist() == [0.0]


@needs_openavl
@pytest.mark.unit
def test_the_lattice_refuses_a_planform_it_cannot_build_on():
    """It needs sections, so a planform that does not describe a trapezoid is refused by name."""

    class Circle(AerodynamicLoads):
        """Not a planform at all -- used only to type-check the refusal."""

        model_name = "circle"

        @classmethod
        def build(cls, *, planform, span_efficiency, zero_lift_drag):
            """Never used."""
            return cls()

        def coefficients(self, condition, planform):
            """Never used."""
            return AeroCoefficients(CL=0.0, CD=0.0)

        def drag_gradients(self, condition, planform):
            """Never used."""
            return dict.fromkeys(self.GRADIENT_NAMES, np.zeros(1))

    with pytest.raises(LoadsError, match="does not describe one"):
        OpenAVLLoads(planform=Circle())


@needs_openavl
@pytest.mark.integration
@pytest.mark.slow
def test_the_lattice_changes_the_aeroplane_the_mission_sizes():
    """The swap must be live: a different aerodynamics must produce a different aeroplane.

    Otherwise the model could be installed and ignored, which every other test here would pass.
    The direction is checked as well as the magnitude -- the lattice finds this wing better than
    the assumed span efficiency, so it must burn less fuel, not merely different fuel.
    """
    polar = SizingAnalysis(_cdadt_case(nodes=11))
    polar.build()
    polar.converge()
    polar_fuel = float(polar.box.get("mission.loiter.fuel_burn_integ.fuel_burn_final", units="kg"))

    lattice = SizingAnalysis(_cdadt_case(loads="cdadt.adapter.avl:OpenAVLLoads", nodes=11))
    lattice.build()
    lattice.converge()
    lattice_fuel = float(lattice.box.get("mission.loiter.fuel_burn_integ.fuel_burn_final", units="kg"))

    assert lattice_fuel != pytest.approx(polar_fuel, rel=1e-6), "the lattice made no difference"
    assert lattice_fuel < polar_fuel, (
        f"the lattice reports e near 0.99 against the case file's 0.82, so it must burn less fuel; "
        f"got {lattice_fuel:.1f} kg against {polar_fuel:.1f} kg"
    )


# =============================================================================================
# The edges: errors, reprs, and the dependency being absent
# =============================================================================================


@pytest.mark.unit
def test_a_drag_polar_that_falls_with_lift_cannot_be_constructed():
    """Induced drag grows with lift. Anything else is a lattice that failed, not a result.

    A negative curvature reaching an optimizer is worse than an error: it becomes an incentive to
    add lift for less drag, and the design walks off into a region the solver invented. Checked on
    the value object rather than through a real lattice, because it is the value object that
    enforces it -- and because reproducing it in the lattice would mean finding a geometry openavl
    cannot resolve.

    Needs no openavl: the invariant belongs to the polar, not to the solver that fills it in.
    """
    good = LatticePolar(
        minimum_drag=0.0,
        curvature=0.034,
        lift_at_minimum_drag=0.0,
        span_efficiency=0.99,
        jacobian=((0.0,) * 4,) * 3,
        describes="a wing of area 124.6 m2",
    )
    assert float(good.drag_at(np.asarray(0.5))) == pytest.approx(0.034 * 0.25)
    # It names the wing and the two numbers a failed run needs: a polar in a traceback that does
    # not say which geometry produced it is not much use.
    assert good.describes == "a wing of area 124.6 m2"
    assert repr(good) == "LatticePolar('a wing of area 124.6 m2', curvature=0.034, e=0.99)"

    with pytest.raises(LoadsError, match="non-positive curvature"):
        LatticePolar(
            minimum_drag=0.0,
            curvature=-1e-3,
            lift_at_minimum_drag=0.0,
            span_efficiency=0.99,
            jacobian=((0.0,) * 4,) * 3,
            describes="an impossible wing",
        )


@needs_openavl
@pytest.mark.unit
def test_a_model_built_on_one_wing_refuses_to_report_another():
    """The lattice describes one geometry. Reusing it silently would report a different aeroplane."""
    built_on = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    something_else = TrapezoidalPlanform(area=90.0, aspect_ratio=12.0, sweep=25.0, taper=0.159)
    loads = OpenAVLLoads(planform=built_on)

    with pytest.raises(LoadsError, match="different wing"):
        loads.coefficients(FlightCondition(CL=0.5, mach=0.0), something_else)


@pytest.mark.unit
def test_the_lattice_model_says_so_when_openavl_is_missing(monkeypatch):
    """It is an optional extra, so its absence must name the extra rather than fail on an import.

    Reached by making the import fail rather than by uninstalling anything: a ``None`` in
    ``sys.modules`` is what the import system treats as an unimportable module. This is the path a
    machine without JAX takes, and it has to give an instruction rather than a traceback.
    """
    import sys

    monkeypatch.setitem(sys.modules, "openavl", None)

    from cdadt.adapter.avl import openavl_is_available as probe

    assert not probe()
    with pytest.raises(LoadsError, match=r'pip install -e "\.\[avl\]"'):
        OpenAVLLoads(planform=TrapezoidalPlanform(area=124.6, aspect_ratio=9.45))


@pytest.mark.unit
def test_the_adapter_objects_repr_as_what_they_configure():
    """These name the phase and the aerodynamics, which is what a failed run needs to say."""
    aircraft = CdadtAircraftModel(num_nodes=11, flight_phase="cruise", loads_factory=PolarLoads)
    assert repr(aircraft) == "CdadtAircraftModel('cruise', num_nodes=11)"

    group = SizingMissionAnalysis(num_nodes=21, aerodynamic_loads="cdadt.models.polar:PolarLoads")
    assert repr(group) == (
        "SizingMissionAnalysis(num_nodes=21, aerodynamic_loads='cdadt.models.polar:PolarLoads', wave_drag=False)"
    )


# =============================================================================================
# The second solver behind the same slot: OpenAeroStruct
# =============================================================================================


@needs_openaerostruct
@pytest.mark.validation
def test_the_openaerostruct_polar_reports_its_own_fit_error():
    """This polar is a fit, not an identity, and the difference is published rather than assumed.

    openavl's quadratic is exact: the Trefftz-plane drag is a quadratic form in a circulation affine
    in angle of attack, so three solves determine it and the residual is round-off. The near-field
    sum OpenAeroStruct publishes carries no such guarantee. Measured rather than trusted, and the
    number is in the failure message so a regression says how much worse it got.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    lattice = OpenAeroStructLattice(wing, 0.3, 6, 20)
    polar = lattice.fit()

    worst = 0.0
    for angle in (2.0, 6.0, 11.0):  # none of them sampled by the fit
        assert angle not in ANGLES_OF_ATTACK_DEG
        lift, drag = lattice.coefficients_at(angle)
        worst = max(worst, abs(float(polar.drag_at(np.asarray(lift))) / drag - 1.0))

    assert worst < 5e-3, f"the OpenAeroStruct polar now drifts from its own lattice by {worst:.2e}"
    # And it is *not* exact, which is the honest difference from the openavl model. If this ever
    # passes, the near-field drag has become quadratic and the documentation is wrong.
    assert worst > 1e-8, "this polar is now exact, which openavl's is and this one should not be"


@needs_openaerostruct
@pytest.mark.validation
def test_the_openaerostruct_geometry_derivatives_are_exact():
    """OpenMDAO already has the derivatives; the work is differentiating the fit on top of them.

    The fit is a determined linear system, so its derivative is another solve with the same matrix
    -- plus a term for the lift at a fixed angle of attack moving when the wing does. Dropping that
    term is the mistake that would make these silently approximate, and differencing the model as
    installed is what catches it.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    condition = FlightCondition(CL=0.5, mach=0.3)
    analytic = OpenAeroStructLoads(planform=wing).drag_gradients(condition, wing)

    def drag_coefficient(area: float, aspect: float, sweep: float, taper: float) -> float:
        """CD from the model as installed, at the same flight condition."""
        moved = TrapezoidalPlanform(area=area, aspect_ratio=aspect, sweep=sweep, taper=taper)
        return float(OpenAeroStructLoads(planform=moved).coefficients(condition, moved).CD[0])

    base = (124.6, 9.45, 25.0, 0.159)
    worst = {}
    for index, (name, step) in enumerate({"area": 0.05, "AR": 0.005, "sweep": 0.02, "taper": 0.0005}.items()):
        moved_up, moved_down = list(base), list(base)
        moved_up[index] += step
        moved_down[index] -= step
        differenced = (drag_coefficient(*moved_up) - drag_coefficient(*moved_down)) / (2.0 * step)
        worst[name] = abs(float(analytic[name][0]) - differenced) / max(abs(differenced), 1e-12)

    assert worst["AR"] < 1e-5, f"dCD/dAR is off by {worst['AR']:.2e}"
    assert worst["sweep"] < 1e-5, f"dCD/dsweep is off by {worst['sweep']:.2e}"
    assert worst["taper"] < 1e-4, f"dCD/dtaper is off by {worst['taper']:.2e}"
    assert abs(float(analytic["area"][0])) < 1e-12, "the drag coefficient should not depend on area"


@needs_openaerostruct
@pytest.mark.unit
def test_each_model_supplies_the_workspace_it_needs():
    """The model says what store it needs; the caller decides how long it lives.

    This is the pairing that stops a case file naming one lattice and being handed the other one's
    answers. The analysis group used to construct the store itself, which silently meant openavl's,
    so the OpenAeroStruct model was refused its own workspace -- correctly, and only because the
    library checks. Now the group asks.
    """
    from cdadt.adapter.lattice import DifferentiableLattice, LatticeLibrary

    assert PolarLoads.new_workspace() is None, "an equation has nothing expensive to keep"

    for model, solver in ((OpenAVLLoads, DifferentiableLattice), (OpenAeroStructLoads, OpenAeroStructLattice)):
        workspace = model.new_workspace()
        assert isinstance(workspace, LatticeLibrary)
        assert workspace.solver is solver, f"{model.__name__} asked for the wrong solver's library"
        assert len(workspace) == 0, "a fresh workspace has solved nothing"
        # And the model accepts the one it asked for, which is the round trip that matters.
        wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
        assert model.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace=workspace)


@needs_openaerostruct
@pytest.mark.unit
def test_a_library_belongs_to_one_solver():
    """A store of solved polars is a store of *one code's* answers, and mixing them is silent.

    The two solvers disagree by about 4% on the same wing, because one reports far-field drag and
    the other near-field. A library filled by openavl handed to this model would report openavl's
    drag under this model's name -- no error, no wrong shape, just the other code's answer.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)

    matching = LatticeLibrary(solver=OpenAeroStructLattice)
    assert (
        OpenAeroStructLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace=matching)
        .polar()
        .curvature
        > 0.0
    )
    assert len(matching) == 1
    assert repr(matching) == "LatticeLibrary(OpenAeroStructLattice, 1 solved)"

    # Offered nothing, it keeps its own -- correct, and the path a standalone study takes.
    alone = OpenAeroStructLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace=None)
    assert alone.polar().curvature == pytest.approx(
        OpenAeroStructLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace=matching)
        .polar()
        .curvature,
        rel=1e-12,
    ), "a private library must give the same answer as a shared one, only slower"

    with pytest.raises(LoadsError, match="A library belongs to one solver"):
        OpenAeroStructLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace=LatticeLibrary())
    with pytest.raises(LoadsError, match="cannot store anything in"):
        OpenAeroStructLoads.build(planform=wing, span_efficiency=0.8, zero_lift_drag=0.0, workspace="nonsense")


@needs_openaerostruct
@pytest.mark.unit
def test_the_case_file_can_name_either_solver():
    """The whole point of the slot: two vortex lattices, one line of configuration apart."""
    for named, expected in (
        ("cdadt.adapter.oas:OpenAeroStructLoads", OpenAeroStructLoads),
        ("cdadt.adapter.avl:OpenAVLLoads", OpenAVLLoads),
        ("cdadt.models.polar:PolarLoads", PolarLoads),
    ):
        group = SizingMissionAnalysis(num_nodes=3, aerodynamic_loads=named)
        assert group._loads_factory() is expected
        assert issubclass(expected, AerodynamicLoads)

    # Distinct names, because a run record has to say which code produced its numbers.
    names = {model.model_name for model in (OpenAeroStructLoads, OpenAVLLoads, PolarLoads)}
    assert len(names) == 3, f"two models share a name: {names}"


@needs_openaerostruct
@pytest.mark.unit
def test_the_openaerostruct_model_refuses_a_wing_it_cannot_mesh_or_was_not_built_on():
    """The two ways a caller can hand this model the wrong aeroplane, both refused by name.

    A mesh needs a trapezoid, and a model built on one wing must not report another. Neither would
    raise on its own: the first would fail later inside OpenConcept on a missing attribute, and the
    second would quietly return a different aircraft's drag.
    """

    class NotAPlanform(Planform):
        """Satisfies the abstraction and describes no trapezoid."""

        @property
        def area(self) -> float:
            """Any number will do."""
            return 100.0

        @property
        def aspect_ratio(self) -> float:
            """Any number will do."""
            return 9.0

    with pytest.raises(LoadsError, match="does not describe one"):
        OpenAeroStructLattice(NotAPlanform(), 0.3, 6, 20)
    with pytest.raises(LoadsError, match="does not describe one"):
        OpenAeroStructLoads(planform=NotAPlanform())

    built_on = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    loads = OpenAeroStructLoads(planform=built_on)
    # The span efficiency this lattice implies -- above 1, because its drag is a near-field sum.
    assert loads.span_efficiency > 1.0

    with pytest.raises(LoadsError, match="different wing"):
        loads.coefficients(FlightCondition(CL=0.5, mach=0.3), TrapezoidalPlanform(area=90.0, aspect_ratio=12.0))
    with pytest.raises(LoadsError, match="different wing"):
        loads.drag_gradients(FlightCondition(CL=0.5, mach=0.3), TrapezoidalPlanform(area=90.0, aspect_ratio=12.0))


@pytest.mark.unit
def test_the_openaerostruct_model_says_so_when_the_extra_is_missing(monkeypatch):
    """It is an optional extra reached through OpenConcept, so its absence must name the extra.

    Reached by making the import fail rather than by uninstalling anything: a ``None`` in
    ``sys.modules`` is what the import system treats as an unimportable module. This is the path a
    machine without OpenAeroStruct takes, and it has to give an instruction rather than a traceback
    from inside somebody else's package.
    """
    import sys

    monkeypatch.setitem(sys.modules, "openconcept.aerodynamics.openaerostruct", None)

    from cdadt.adapter.oas import openaerostruct_is_available as probe

    assert not probe()
    with pytest.raises(LoadsError, match=r'pip install -e "\.\[transonic\]"'):
        OpenAeroStructLattice(TrapezoidalPlanform(area=124.6, aspect_ratio=9.45), 0.3, 6, 20)
