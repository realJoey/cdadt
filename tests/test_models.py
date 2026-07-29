"""Unit and integration: the physics cdadt owns.

These are the first components in the package that *compute* something, so they are tested
differently from the rest of cdadt. Two claims matter and neither is about plumbing: the polar
reproduces the equation OpenConcept evaluates, and the geometry it is evaluated on is the wing
the case file describes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdadt.models import (
    AeroCoefficients,
    AerodynamicLoads,
    FlightCondition,
    LoadsError,
    PolarLoads,
    TrapezoidalPlanform,
)
from cdadt.models.planform import WingSection

# =============================================================================================
# Coefficients
# =============================================================================================


@pytest.mark.unit
def test_coefficients_broadcast_a_single_value_to_every_point():
    """A model that produces one constant moment over a phase should not have to say so N times."""
    coefficients = AeroCoefficients(CL=[0.4, 0.5, 0.6], CD=0.02)

    assert len(coefficients) == 3
    assert coefficients.CD.tolist() == [0.02, 0.02, 0.02]
    assert coefficients.CL.tolist() == [0.4, 0.5, 0.6]


@pytest.mark.unit
def test_components_given_at_different_points_are_refused():
    """Drag at every node and a moment at two is a bug that would otherwise become a broadcast."""
    with pytest.raises(ValueError, match=r"'Cm' has 2 values but the coefficients have 3"):
        AeroCoefficients(CL=[0.4, 0.5, 0.6], CD=[0.02, 0.02, 0.02], Cm=[0.0, 0.1])


@pytest.mark.unit
def test_symmetric_flight_is_distinguished_from_a_model_that_has_no_moments():
    """Zero sideforce is an answer; the flag says which reading applies."""
    symmetric = AeroCoefficients(CL=0.5, CD=0.02)
    yawed = AeroCoefficients(CL=0.5, CD=0.02, CY=0.03, Cn=-0.01)

    assert not symmetric.lateral_directional
    assert yawed.lateral_directional
    assert repr(symmetric) == "AeroCoefficients(1 points, symmetric)"
    assert repr(yawed) == "AeroCoefficients(1 points, 6-component)"


@pytest.mark.unit
def test_a_component_can_be_selected_by_name_and_an_unknown_one_is_refused():
    """A case file may name which coefficient a constraint is written against."""
    coefficients = AeroCoefficients(CL=0.5, CD=0.02)

    assert coefficients["CD"].tolist() == [0.02]
    assert [name for name, _ in coefficients] == list(AeroCoefficients.NAMES)
    with pytest.raises(KeyError, match="not an aerodynamic coefficient"):
        coefficients["CDi"]


# =============================================================================================
# Flight condition
# =============================================================================================


@pytest.mark.unit
def test_a_flight_condition_broadcasts_and_refuses_mismatched_lengths():
    """A whole phase at one altitude is ordinary; a phase with two Machs and three CLs is not."""
    condition = FlightCondition(CL=[0.4, 0.5], mach=0.78)

    assert len(condition) == 2
    assert condition.mach.tolist() == [0.78, 0.78]
    assert repr(condition) == "FlightCondition(2 points)"

    # The point count is the longest quantity given, so it is the *shorter* one that is named.
    # Which of the two the caller got wrong is not knowable here; what matters is that the
    # mismatch is refused rather than broadcast into a mission.
    with pytest.raises(ValueError, match="'CL' has 2 values but the flight condition has 3"):
        FlightCondition(CL=[0.4, 0.5], mach=[0.7, 0.75, 0.8])


# =============================================================================================
# Planform
# =============================================================================================


@pytest.mark.unit
def test_the_planform_reproduces_the_shipped_wing():
    """Checked against the closed-form trapezoid relations, not against a previous run."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)

    assert wing.span == pytest.approx(np.sqrt(9.45 * 124.6))
    # S = b * c_root * (1 + taper) / 2, rearranged.
    assert wing.root_chord == pytest.approx(2 * 124.6 / (wing.span * 1.159))
    assert wing.tip_chord == pytest.approx(wing.root_chord * 0.159)
    assert wing.area == 124.6 and wing.aspect_ratio == 9.45


@pytest.mark.unit
def test_the_mean_aerodynamic_chord_matches_the_standard_relation():
    """An untapered wing's MAC is its chord; the general case follows the trapezoid formula."""
    square = TrapezoidalPlanform(area=100.0, aspect_ratio=10.0, taper=1.0)
    assert square.mean_aerodynamic_chord == pytest.approx(square.root_chord)

    tapered = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, taper=0.159)
    expected = (2 / 3) * tapered.root_chord * (1 + 0.159 + 0.159**2) / (1 + 0.159)
    assert tapered.mean_aerodynamic_chord == pytest.approx(expected)


@pytest.mark.unit
def test_the_tip_is_placed_so_the_quarter_chord_carries_the_stated_sweep():
    """``c4sweep`` means the quarter chord. Sweeping the leading edge is a different wing."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)
    root, tip = wing.sections()

    root_quarter_chord = root.x + 0.25 * root.chord
    tip_quarter_chord = tip.x + 0.25 * tip.chord
    swept = np.degrees(np.arctan2(tip_quarter_chord - root_quarter_chord, tip.y - root.y))

    assert swept == pytest.approx(25.0)
    assert tip.y == pytest.approx(0.5 * wing.span), "sections describe the starboard semi-span"


@pytest.mark.unit
def test_an_unbuildable_wing_is_refused_rather_than_producing_a_lattice_of_nothing():
    """Each of these would otherwise surface as a divide-by-zero somewhere inside a solver."""
    with pytest.raises(ValueError, match="positive reference area"):
        TrapezoidalPlanform(area=0.0, aspect_ratio=9.45)
    with pytest.raises(ValueError, match="positive aspect ratio"):
        TrapezoidalPlanform(area=124.6, aspect_ratio=-1.0)
    with pytest.raises(ValueError, match="must be positive"):
        TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, taper=0.0)
    with pytest.raises(ValueError, match="positive chord"):
        WingSection(x=0.0, y=0.0, z=0.0, chord=0.0)


@pytest.mark.unit
def test_a_taper_above_one_is_allowed_because_a_derivative_has_to_cross_it():
    """Inverse taper is unusual, not unbuildable -- and capping it broke finite differences.

    A finite difference at ``taper = 1`` steps to 1.000001. An earlier version rejected that,
    which made the class non-differentiable at exactly the bound an optimizer is most likely to
    sit on. Bounds on a design variable belong in the case file's ``optimize`` band.
    """
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, taper=1.000001)
    assert wing.tip_chord > wing.root_chord


# =============================================================================================
# The loads abstraction
# =============================================================================================


@pytest.mark.unit
def test_a_model_must_name_itself_when_it_is_written():
    """The run record says which aerodynamics produced its numbers, so the name cannot default."""
    with pytest.raises(TypeError, match="must declare a 'model_name'"):

        class Nameless(AerodynamicLoads):
            """A model that forgot the one thing it cannot inherit."""

            def coefficients(self, condition, planform):
                """Never reached."""
                return AeroCoefficients(CL=0.0, CD=0.0)


@pytest.mark.unit
def test_the_abstraction_cannot_be_instantiated_without_producing_coefficients():
    """``coefficients`` is the whole contract."""
    with pytest.raises(TypeError, match="abstract"):
        AerodynamicLoads()


# =============================================================================================
# The parabolic polar
# =============================================================================================


@pytest.mark.unit
def test_the_polar_is_the_textbook_equation():
    """CD = CD0 + CL^2 / (pi e AR), evaluated rather than recalled."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45)
    loads = PolarLoads(span_efficiency=0.82, zero_lift_drag=0.02)

    condition = FlightCondition(CL=[0.4, 0.5, 0.6], mach=0.78)
    expected = 0.02 + np.array([0.4, 0.5, 0.6]) ** 2 / (np.pi * 0.82 * 9.45)

    assert pytest.approx(expected) == loads.coefficients(condition, wing).CD
    assert loads.model_name == "parabolic_polar"


@pytest.mark.unit
def test_the_polar_reports_no_moments_and_says_so():
    """A parabolic polar carries no moment information; a fabricated zero would be a claim."""
    coefficients = PolarLoads(span_efficiency=0.82).coefficients(
        FlightCondition(CL=0.5, mach=0.78), TrapezoidalPlanform(area=124.6, aspect_ratio=9.45)
    )
    assert not coefficients.lateral_directional
    assert coefficients.Cm.tolist() == [0.0]


@pytest.mark.unit
def test_drag_is_the_coefficient_times_dynamic_pressure_and_area():
    """The force the trajectory consumes, derived from the coefficient rather than alongside it."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45)
    loads = PolarLoads(span_efficiency=0.82, zero_lift_drag=0.02)
    condition = FlightCondition(CL=0.5, mach=0.78, dynamic_pressure=1.2e4)

    coefficients = loads.coefficients(condition, wing)
    assert loads.drag(condition, wing) == pytest.approx(coefficients.CD * 1.2e4 * 124.6)


@pytest.mark.unit
def test_a_zero_span_efficiency_is_refused_as_the_division_it_is():
    """Zero e is not 'no induced drag'."""
    with pytest.raises(LoadsError, match="span efficiency must be positive"):
        PolarLoads(span_efficiency=0.0)


@pytest.mark.unit
def test_the_analytic_drag_gradients_match_finite_differences():
    """These are what an optimizer steps on, so they are checked rather than trusted."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45)
    lift, efficiency, zero_lift, aspect = 0.5, 0.82, 0.02, 9.45
    step = 1e-7

    def drag_of(CL=lift, e=efficiency, CD0=zero_lift, AR=aspect):
        """CD at a perturbed operating point, through the model rather than the formula."""
        model = PolarLoads(span_efficiency=e, zero_lift_drag=CD0)
        return float(
            model.coefficients(FlightCondition(CL=CL, mach=0.78), TrapezoidalPlanform(area=124.6, aspect_ratio=AR)).CD
        )

    gradients = PolarLoads(span_efficiency=efficiency, zero_lift_drag=zero_lift).drag_gradients(
        FlightCondition(CL=lift, mach=0.78), wing
    )

    assert float(gradients["CL"]) == pytest.approx((drag_of(CL=lift + step) - drag_of()) / step, rel=1e-5)
    assert float(gradients["e"]) == pytest.approx((drag_of(e=efficiency + step) - drag_of()) / step, rel=1e-5)
    assert float(gradients["AR"]) == pytest.approx((drag_of(AR=aspect + step) - drag_of()) / step, rel=1e-5)
    assert float(gradients["CD0"]) == pytest.approx(1.0)


# =============================================================================================
# Against OpenConcept's own component
# =============================================================================================


@pytest.mark.integration
def test_the_polar_reproduces_openconcepts_polar_drag_exactly():
    """The claim the whole parity strategy rests on.

    Not "close enough": the same equation evaluated by two implementations must agree to
    round-off. Any looser and a later difference could not be attributed to the new physics
    rather than to this model.
    """
    import openmdao.api as om
    from openconcept.aerodynamics import PolarDrag

    nodes = 5
    lift = np.linspace(0.35, 0.65, nodes)
    pressure = np.linspace(9.0e3, 1.4e4, nodes)
    area, aspect, efficiency, zero_lift = 124.6, 9.45, 0.82, 0.019

    problem = om.Problem(reports=False)
    problem.model.add_subsystem("polar", PolarDrag(num_nodes=nodes, vec_CD0=True), promotes=["*"])
    problem.setup(force_alloc_complex=True)
    problem.set_val("fltcond|CL", lift)
    problem.set_val("fltcond|q", pressure, units="N/m**2")
    problem.set_val("ac|geom|wing|S_ref", area, units="m**2")
    problem.set_val("ac|geom|wing|AR", aspect)
    problem.set_val("e", efficiency)
    problem.set_val("CD0", np.full(nodes, zero_lift))
    problem.run_model()

    reference = problem.get_val("drag", units="N")

    loads = PolarLoads(span_efficiency=efficiency, zero_lift_drag=np.full(nodes, zero_lift))
    condition = FlightCondition(CL=lift, mach=0.78, dynamic_pressure=pressure)
    computed = loads.drag(condition, TrapezoidalPlanform(area=area, aspect_ratio=aspect))

    assert computed == pytest.approx(reference, rel=1e-12)


# =============================================================================================
# The parts a report reads
# =============================================================================================


@pytest.mark.unit
def test_every_coefficient_and_condition_component_is_readable():
    """The six components and the four conditions are the interface; all of them are read.

    Not busywork: the lateral-directional four exist so a later trim or handling-qualities
    constraint can be written against them, and an accessor nothing ever calls is an accessor
    nobody has checked returns the right array.
    """
    coefficients = AeroCoefficients(CL=0.5, CD=0.02, CY=0.03, Cl=-0.01, Cm=-0.4, Cn=0.002)

    assert coefficients.CY.tolist() == [0.03]
    assert coefficients.Cl.tolist() == [-0.01]
    assert coefficients.Cm.tolist() == [-0.4]
    assert coefficients.Cn.tolist() == [0.002]

    condition = FlightCondition(CL=0.5, mach=0.78, altitude=10668.0, dynamic_pressure=1.2e4)
    assert condition.altitude.tolist() == [10668.0]
    assert condition.dynamic_pressure.tolist() == [1.2e4]


@pytest.mark.unit
def test_the_polar_reports_what_it_was_built_from():
    """A run record says which aerodynamics produced its numbers, and on what assumptions."""
    loads = PolarLoads(span_efficiency=0.82, zero_lift_drag=0.019)

    assert loads.span_efficiency == 0.82
    assert loads.zero_lift_drag.tolist() == [0.019]
    assert repr(loads) == "PolarLoads('parabolic_polar')"


@pytest.mark.unit
def test_the_geometry_objects_repr_as_the_wing_they_describe():
    """These appear in tracebacks from a failed lattice solve, so they must name the wing."""
    wing = TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25.0, taper=0.159)

    assert repr(wing) == "TrapezoidalPlanform(area=124.6, aspect_ratio=9.45, sweep=25, taper=0.159)"
    root, _tip = wing.sections()
    assert repr(root) == f"WingSection(y={root.y:.3f}, chord={root.chord:.3f})"
