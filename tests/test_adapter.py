"""Integration: cdadt's own aerodynamics installed in OpenConcept's mission.

The claims here are the ones the substitution rests on. The analysis group reproduces the
reference exactly with the parity model, so the wiring is proven; the loads model is genuinely
consulted, so the swap is not decorative; and the vortex lattice produces a physically coherent
aeroplane, checked against what theory says rather than against a stored number.
"""

from __future__ import annotations

import copy

import numpy as np
import openmdao.api as om
import pytest

from cdadt import Config, SizingAnalysis
from cdadt.adapter.aircraft import CdadtAircraftModel
from cdadt.adapter.analysis import SizingMissionAnalysis, jet_transport_zero_lift_drag
from cdadt.adapter.avl import OpenAVLLoads, openavl_is_available
from cdadt.adapter.loads import AerodynamicLoadsComp
from cdadt.models import AeroCoefficients, AerodynamicLoads, FlightCondition, LoadsError, PolarLoads
from cdadt.models.planform import TrapezoidalPlanform
from tests.conftest import case_dict

#: Skip marker for everything that needs the optional vortex-lattice extra.
needs_openavl = pytest.mark.skipif(not openavl_is_available(), reason="openavl is not installed")

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
    assert problem.get_val("aircraft.zero_lift_drag.CD0").tolist() == [0.0, 0.0, 0.0]
    assert problem.get_val("drag", units="N") == pytest.approx(np.full(3, expected), rel=1e-10)


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
@pytest.mark.unit
def test_the_closed_form_polar_matches_the_lattice_at_lift_it_never_sampled():
    """The fit is three solves; this checks it against the lattice where it was not fitted.

    Measured rather than asserted at a chosen tolerance: the error is reported in the message so
    a regression says how much worse it got, not merely that it did.
    """
    from openavl import Aircraft, AVLSolver

    from cdadt.adapter.avl import _lattice

    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    loads = OpenAVLLoads(planform=wing)
    aircraft = _lattice(wing, 0.0, 6, 20, Aircraft)

    worst = 0.0
    for lift in (0.35, 0.45, 0.65):
        solver = AVLSolver(aircraft, cd0=0.0, rho=1.225)
        solver.set_parameter("cl", lift)
        solver.setup_trim(mode=1)
        solver.execute_run(max_iter=30)
        direct = float(solver.get_results()["CDFF"])
        fitted = float(np.atleast_1d(loads.coefficients(FlightCondition(CL=lift, mach=0.0), wing).CD)[0])
        worst = max(worst, abs(fitted - direct) / direct)

    assert worst < 5e-3, f"the fitted polar drifts from the lattice by {worst:.2e} over the mission range"


@needs_openavl
@pytest.mark.unit
def test_the_neglected_span_efficiency_gradient_stays_small():
    """The AR derivative holds the lattice's span efficiency fixed. This measures what that costs.

    The exact derivative carries a term in ``de/dAR`` that this model does not include. It is
    about 1.8% of the term that is kept. Measured here so that a change in the lattice, the
    resolution or the fit cannot quietly turn a good approximation into a bad one.
    """
    from cdadt.adapter.avl import _lattice_span_efficiency

    area, aspect, sweep, taper, mach = 124.6, 9.45, 25.0, 0.159, 0.0
    step = 0.05
    low = _lattice_span_efficiency(area, aspect - step, sweep, taper, mach, 6, 20)
    high = _lattice_span_efficiency(area, aspect + step, sweep, taper, mach, 6, 20)
    middle = _lattice_span_efficiency(area, aspect, sweep, taper, mach, 6, 20)

    neglected_over_retained = abs((aspect / middle) * (high - low) / (2 * step))
    assert neglected_over_retained < 0.05, (
        f"holding the span efficiency fixed now misses {neglected_over_retained:.1%} of dCD/dAR, "
        f"which is no longer a small correction"
    )


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
def test_a_drag_polar_that_falls_with_lift_is_refused():
    """Induced drag grows with lift. Anything else is a lattice that failed, not a result.

    A negative curvature reaching an optimizer is worse than an error: it becomes an incentive to
    add lift for less drag, and the design walks off into a region the solver invented. Tested on
    the fit itself, which is why it is a separate function -- reproducing it through a real
    lattice would mean finding a geometry AVL cannot resolve.
    """
    from cdadt.adapter.avl import quadratic_polar

    # Well-behaved first, and checked by the property that matters: three points determine a
    # quadratic exactly, so the fit must reproduce the points it was given.
    lift = [0.2, 0.5, 0.8]
    drag = [0.004, 0.010, 0.022]
    cd_min, curvature, cl_at_min = quadratic_polar(lift, drag)

    assert curvature > 0.0, "induced drag must grow with lift"
    for sampled_lift, sampled_drag in zip(lift, drag, strict=True):
        assert cd_min + curvature * (sampled_lift - cl_at_min) ** 2 == pytest.approx(sampled_drag, rel=1e-10)

    # Now a set that genuinely curves the wrong way. Simply reversing the drags is not enough --
    # a decreasing but decelerating sequence still fits an upward parabola whose vertex lies
    # beyond the samples, and its curvature is positive. What is unphysical is *concave* drag:
    # rising with lift but flattening, so the second difference is negative.
    with pytest.raises(LoadsError, match="non-positive curvature"):
        quadratic_polar([0.2, 0.5, 0.8], [0.004, 0.012, 0.014], describes="an impossible wing")


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
    assert repr(group) == ("SizingMissionAnalysis(num_nodes=21, aerodynamic_loads='cdadt.models.polar:PolarLoads')")
