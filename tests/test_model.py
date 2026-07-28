"""Integration tests for the disciplines and the assembled model.

These build real OpenMDAO models and, where the claim is about a number, run them. They do not
converge the full mission -- that is :mod:`tests.test_validation_b738`.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt import JetTransportPhaseModel, Part25PhaseModel, SizingModel
from cdadt.disciplines import Aerodynamics, Geometry, HighLift, MassProperties, Propulsion, Stability, Weights

pytestmark = pytest.mark.integration


# ======================================================================================
# Disciplines
# ======================================================================================


def test_takeoff_configuration_is_decided_by_the_phase():
    """High-lift drag belongs to the takeoff phases and to no others."""
    for phase in ("v0v1", "v1v0", "v1vr", "rotate"):
        assert Aerodynamics(num_nodes=3, flight_phase=phase).in_takeoff_configuration
    for phase in ("climb", "cruise", "descent", "loiter", "EngineOutClimbAngle"):
        assert not Aerodynamics(num_nodes=3, flight_phase=phase).in_takeoff_configuration


def test_part25_aerodynamics_adds_the_engine_out_climb_condition():
    """25.121(b) specifies takeoff flaps for the second-segment climb."""
    from cdadt.model import Part25Aerodynamics

    assert Part25Aerodynamics(num_nodes=1, flight_phase="EngineOutClimbAngle").in_takeoff_configuration
    # It must not disturb the phases the base class already covers.
    assert Part25Aerodynamics(num_nodes=1, flight_phase="cruise").in_takeoff_configuration is False


def test_geometry_computes_the_mac_and_the_lever_arms():
    """Both tails read the same fraction of fuselage length, and the MAC is the trapezoidal one."""
    problem = om.Problem(model=Geometry(), reports=False)
    problem.setup(check=False)
    problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    problem.set_val("ac|geom|wing|AR", 9.45)
    problem.set_val("ac|geom|wing|taper", 0.159)
    problem.set_val("ac|geom|fuselage|length", 38.08, units="m")
    problem.run_model()

    arm = problem.get_val("ac|geom|hstab|c4_to_wing_c4", units="m").item()
    assert arm == pytest.approx(0.5 * 38.08)
    assert problem.get_val("ac|geom|vstab|c4_to_wing_c4", units="m").item() == pytest.approx(arm)

    # Trapezoidal MAC, computed independently of the component under test.
    span = np.sqrt(9.45 * 124.6)
    root = 2 * 124.6 / (span * (1 + 0.159))
    expected = (2 / 3) * root * (1 + 0.159 + 0.159**2) / (1 + 0.159)
    assert problem.get_val("ac|geom|wing|MAC", units="m").item() == pytest.approx(expected, rel=1e-9)


def test_stability_sizes_both_tails_from_volume_coefficients():
    """Raymer's tail volume coefficient relations, checked against the formulas."""
    problem = om.Problem(model=Stability(), reports=False)
    problem.setup(check=False)
    problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    problem.set_val("ac|geom|wing|AR", 9.45)
    problem.set_val("ac|geom|wing|MAC", 4.2684, units="m")
    problem.set_val("ac|geom|hstab|c4_to_wing_c4", 19.04, units="m")
    problem.set_val("ac|geom|vstab|c4_to_wing_c4", 19.04, units="m")
    problem.run_model()

    assert problem.get_val("ac|geom|hstab|S_ref", units="m**2").item() == pytest.approx(
        Stability.horizontal_tail_volume_coefficient * 4.2684 * 124.6 / 19.04, rel=1e-6
    )
    assert problem.get_val("ac|geom|vstab|S_ref", units="m**2").item() == pytest.approx(
        Stability.vertical_tail_volume_coefficient * np.sqrt(9.45) * 124.6**1.5 / 19.04, rel=1e-6
    )


def test_high_lift_flaps_add_to_the_clean_maximum():
    """The flapped maximum must exceed the clean one it is built from."""
    problem = om.Problem(model=HighLift(), reports=False)
    problem.setup(check=False)
    problem.set_val("ac|aero|airfoil_Cl_max", 1.75)
    problem.set_val("ac|geom|wing|c4sweep", 25.0, units="deg")
    problem.set_val("ac|geom|wing|toverc", 0.12)
    problem.set_val("ac|aero|takeoff_flap_deg", 15.0, units="deg")
    problem.run_model()

    clean = problem.get_val("ac|aero|CLmax_cruise").item()
    takeoff = problem.get_val("ac|aero|CLmax_TO").item()
    assert 0.0 < clean < 1.75  # sweep can only reduce the section maximum
    assert takeoff > clean


def test_mass_integrates_fuel_flow_and_subtracts_it_from_mtow():
    """Constant fuel flow over a known duration gives a known burn and a falling weight."""
    problem = om.Problem(model=MassProperties(num_nodes=5, flight_phase="cruise"), reports=False)
    problem.setup(check=False)
    problem.set_val("fuel_flow", np.full(5, 0.5), units="kg/s")
    problem.set_val("fuel_burn_integ.duration", 100.0, units="s")
    problem.set_val("ac|weights|MTOW", 70000.0, units="kg")
    problem.run_model()

    assert problem.get_val("fuel_burn_final", units="kg").item() == pytest.approx(50.0)
    assert problem.get_val("weight", units="kg")[-1] == pytest.approx(70000.0 - 50.0)


def test_propulsion_scales_with_running_engines():
    """One failed engine of two must halve thrust and fuel flow."""
    results = {}
    for propulsor_active in (1.0, 0.0):
        problem = om.Problem(model=Propulsion(num_nodes=3, flight_phase="climb"), reports=False)
        problem.setup(check=False)
        problem.set_val("ac|propulsion|num_engines", 2.0)
        problem.set_val("ac|propulsion|engine|rating", 27000.0, units="lbf")
        problem.set_val("throttle", np.full(3, 0.9))
        problem.set_val("fltcond|h", np.full(3, 10000.0), units="ft")
        problem.set_val("fltcond|M", np.full(3, 0.5))
        problem.set_val("propulsor_active", np.full(3, propulsor_active))
        problem.run_model()
        results[propulsor_active] = (
            problem.get_val("thrust", units="lbf")[0],
            problem.get_val("fuel_flow", units="kg/s")[0],
        )

    both, one = results[1.0], results[0.0]
    assert one[0] == pytest.approx(both[0] / 2.0, rel=1e-9)
    assert one[1] == pytest.approx(both[1] / 2.0, rel=1e-9)


def test_aerodynamics_produces_more_drag_with_flaps_out():
    """The takeoff buildup adds flap and gear drag; at the same condition it must cost more."""
    drag = {}
    for phase in ("cruise", "v0v1"):
        problem = om.Problem(model=Aerodynamics(num_nodes=1, flight_phase=phase), reports=False)
        problem.setup(check=False)
        problem.set_val("fltcond|Utrue", [80.0], units="m/s")
        problem.set_val("fltcond|rho", [1.225], units="kg/m**3")
        problem.set_val("fltcond|T", [288.15], units="K")
        problem.set_val("fltcond|q", [0.5 * 1.225 * 80.0**2], units="Pa")
        problem.set_val("fltcond|CL", [0.5])
        problem.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
        problem.set_val("ac|geom|wing|AR", 9.45)
        problem.run_model()
        drag[phase] = problem.get_val("drag", units="N")[0]

    assert drag["v0v1"] > drag["cruise"]


# ======================================================================================
# Assembled models
# ======================================================================================


def _promoted_outputs(problem: om.Problem) -> set[str]:
    """Return every promoted output name at the top of a set-up problem's model."""
    problem.final_setup()
    return {
        meta["prom_name"]
        for _, meta in problem.model.list_outputs(out_stream=None, prom_name=True, val=False, return_format="list")
    }


def test_the_phase_model_produces_openconcepts_three_required_outputs():
    """thrust, drag and weight are the whole contract OpenConcept imposes."""
    problem = om.Problem(model=JetTransportPhaseModel(num_nodes=3, flight_phase="cruise"), reports=False)
    problem.setup(check=False)
    promoted = _promoted_outputs(problem)
    assert {"thrust", "drag", "weight", "fuel_flow", "fuel_burn_final"} <= promoted


def test_the_sizing_model_builds_and_publishes_every_design_parameter(aircraft):
    """Each independent parameter reaches the top of the model as its own variable."""
    problem = om.Problem(model=SizingModel(aircraft=aircraft, num_nodes=3), reports=False)
    problem.setup(check=False)

    promoted = _promoted_outputs(problem)
    for name in aircraft.names:
        assert name in promoted, f"{name} is not published by the model"

    # Quantities the model computes must be outputs too, not inputs left at a placeholder.
    for computed in (
        "ac|weights|MTOW",
        "ac|weights|OEW",
        "ac|weights|MLW",
        "ac|geom|wing|MAC",
        "ac|geom|hstab|S_ref",
        "ac|geom|vstab|S_ref",
        "ac|aero|CLmax_cruise",
        "ac|aero|CLmax_TO",
    ):
        assert computed in promoted


def test_the_two_models_between_them_cover_every_discipline():
    """A discipline that is written but never built would be tested and never used."""
    built = set(SizingModel.disciplines) | set(JetTransportPhaseModel.disciplines)
    assert built == {Geometry, Stability, HighLift, Weights, Aerodynamics, Propulsion, MassProperties}
    # Subsystem names must be unique within each model, or one would silently replace another.
    for model in (SizingModel, JetTransportPhaseModel, Part25PhaseModel):
        names = [discipline.discipline_name for discipline in model.disciplines]
        assert len(names) == len(set(names))


def test_an_even_node_count_is_rejected(aircraft):
    """Simpson's rule needs an odd number of points; a silent wrong integral is worse."""
    problem = om.Problem(model=SizingModel(aircraft=aircraft, num_nodes=4), reports=False)
    with pytest.raises(Exception, match="odd"):
        problem.setup(check=False)


def test_the_part25_phase_model_differs_only_in_aerodynamics():
    """The opt-in model must change one discipline and nothing else."""
    base, part25 = JetTransportPhaseModel.disciplines, Part25PhaseModel.disciplines
    assert len(base) == len(part25)
    assert base[1:] == part25[1:]
    assert issubclass(part25[0], base[0])
