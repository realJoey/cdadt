"""Validation of the B738 sizing analysis.

Claim classes: ``validation`` for the comparison against OpenConcept's assembly,
``integration``/``regression`` for the converged sizing result.

What this file covers
---------------------
The **aircraft-level** model: the empty-weight buildup, tail sizing, mean aerodynamic chord,
wetted areas and maximum lift coefficients. These are compared against an assembly built
from the same OpenConcept components, wired as ``openconcept/examples/B738_sizing.py`` wires
them (lines 244-378), and evaluated at a fixed takeoff weight so no solver is involved.

Component-level agreement does not prove the assembly is right: a wrong moment arm or a
dropped connection lives *between* components, not inside them. So this comparison exists
alongside the per-component equivalence tests in ``cdadt/tests/providers/``, not instead of
them.

The **mission-level** results -- fuel burn, balanced field length, V1 -- are validated in
``test_against_openconcept.py``, against OpenConcept's published golden values and against a
live run of its example. The converged sizing result is additionally pinned here as a
regression, which is a weaker claim: it says the answer has not changed, not that it is
right.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest
from openconcept.aerodynamics import CleanCLmax, FlapCLmax
from openconcept.geometry import CylinderSurfaceArea, WingMACTrapezoidal
from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing
from openconcept.utilities import AddSubtractComp
from openconcept.weights import JetTransportEmptyWeight

from cdadt.aircraft import jet_transport_disciplines
from cdadt.core.discipline import DisciplineGroup, DisciplineScope
from cdadt.mission import SizingLoop

ENGINE_DECK = "CFM56"

#: Takeoff and fuel weights the fixed-point comparison is evaluated at. Any values work --
#: the point is that both sides see the same ones -- so these are the configured sizing
#: starting points rather than new numbers.
FIXED_POINT_MTOW_KG = 79000.0
FIXED_POINT_FUEL_KG = 20000.0


# ==============================================================================
# Aircraft-level validation against OpenConcept's own assembly
# ==============================================================================
class _OpenConceptAircraftLevel(om.Group):
    """The non-mission half of OpenConcept's ``B738_sizing`` example.

    Transcribed from ``openconcept/examples/B738_sizing.py`` lines 244-378: the same
    components, the same wiring, the same tail-arm estimate of half the fuselage length.
    Takeoff and fuel weights are inputs here rather than outputs of the sizing loop, so the
    whole chain evaluates once with no solver.
    """

    def setup(self):
        self.add_subsystem(
            "tail_lever_arm_estimate",
            AddSubtractComp(
                output_name="c4_to_wing_c4",
                input_names=["fuselage_length"],
                units="m",
                scaling_factors=[0.5],
            ),
            promotes_inputs=[("fuselage_length", "ac|geom|fuselage|length")],
        )
        self.connect(
            "tail_lever_arm_estimate.c4_to_wing_c4",
            ["ac|geom|hstab|c4_to_wing_c4", "ac|geom|vstab|c4_to_wing_c4"],
        )

        self.add_subsystem(
            "wing_MAC",
            WingMACTrapezoidal(),
            promotes_inputs=[
                ("S_ref", "ac|geom|wing|S_ref"),
                ("AR", "ac|geom|wing|AR"),
                ("taper", "ac|geom|wing|taper"),
            ],
            promotes_outputs=[("MAC", "ac|geom|wing|MAC")],
        )
        self.add_subsystem(
            "vstab_area",
            VStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|AR", "ac|geom|vstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|vstab|S_ref"],
        )
        self.add_subsystem(
            "hstab_area",
            HStabVolumeCoefficientSizing(),
            promotes_inputs=["ac|geom|wing|S_ref", "ac|geom|wing|MAC", "ac|geom|hstab|c4_to_wing_c4"],
            promotes_outputs=["ac|geom|hstab|S_ref"],
        )
        self.add_subsystem(
            "nacelle_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|nacelle|length"), ("D", "ac|geom|nacelle|diameter")],
            promotes_outputs=[("A", "ac|geom|nacelle|S_wet")],
        )
        self.add_subsystem(
            "fuselage_wetted_area",
            CylinderSurfaceArea(),
            promotes_inputs=[("L", "ac|geom|fuselage|length"), ("D", "ac|geom|fuselage|height")],
            promotes_outputs=[("A", "ac|geom|fuselage|S_wet")],
        )
        self.add_subsystem(
            "MLW_calc",
            AddSubtractComp(
                output_name="ac|weights|MLW",
                input_names=["ac|weights|MTOW"],
                units="kg",
                scaling_factors=[0.8],
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["ac|weights|MLW"],
        )
        self.add_subsystem(
            "empty_weight",
            JetTransportEmptyWeight(),
            promotes_inputs=["*"],
            promotes_outputs=[("OEW", "ac|weights|OEW")],
        )
        self.add_subsystem(
            "CL_max_cruise",
            CleanCLmax(),
            promotes_inputs=["ac|aero|airfoil_Cl_max", "ac|geom|wing|c4sweep"],
            promotes_outputs=[("CL_max_clean", "ac|aero|CLmax_cruise")],
        )
        self.add_subsystem(
            "CL_max_takeoff",
            FlapCLmax(),
            promotes_inputs=[
                ("flap_extension", "ac|aero|takeoff_flap_deg"),
                "ac|geom|wing|c4sweep",
                "ac|geom|wing|toverc",
                ("CL_max_clean", "ac|aero|CLmax_cruise"),
            ],
            promotes_outputs=[("CL_max_flap", "ac|aero|CLmax_TO")],
        )

    def configure(self):
        """Resolve the unit disagreements between OpenConcept's own components.

        These components legitimately disagree -- the empty-weight buildup works in ft and
        lbm, the geometry components in m and kg -- so OpenMDAO requires both a unit and a
        value to be declared for each shared input. The placeholder value is irrelevant:
        every one of these is overwritten by ``set_val`` before the model is run, and the
        comparison sets identical values on both sides.
        """
        for name, units in [
            ("ac|geom|wing|S_ref", "m**2"),
            ("ac|geom|fuselage|length", "m"),
            ("ac|geom|fuselage|height", "m"),
            ("ac|weights|MTOW", "kg"),
        ]:
            self.set_input_defaults(name, val=1.0, units=units)


COMPARED_QUANTITIES = [
    ("ac|geom|wing|MAC", "m"),
    ("ac|geom|fuselage|S_wet", "m**2"),
    ("ac|geom|nacelle|S_wet", "m**2"),
    ("ac|geom|hstab|S_ref", "m**2"),
    ("ac|geom|vstab|S_ref", "m**2"),
    ("ac|weights|MLW", "kg"),
    ("ac|weights|OEW", "kg"),
    ("ac|aero|CLmax_cruise", None),
    ("ac|aero|CLmax_TO", None),
]


def _design_inputs(config, names):
    """Return ``(value, units)`` for each name, read from the configuration."""
    return {name: (config.value(name), config.units(name)) for name in names}


@pytest.fixture(scope="module")
def aircraft_level_comparison(b738_config):
    """Evaluate cdadt's and OpenConcept's aircraft-level assemblies at the same point."""
    disciplines = jet_transport_disciplines(b738_config, engine_deck=ENGINE_DECK)
    aircraft = [d for d in disciplines if d.scope is DisciplineScope.AIRCRAFT]

    required = set()
    provided = set()
    for discipline in aircraft:
        required |= {n for n in discipline.requires().names if n.startswith("ac|")}
        provided |= discipline.provides().names
    inputs = _design_inputs(b738_config, sorted(n for n in required - provided if n in b738_config))
    inputs["ac|weights|MTOW"] = (np.array([FIXED_POINT_MTOW_KG]), "kg")
    inputs["ac|weights|W_fuel_max"] = (np.array([FIXED_POINT_FUEL_KG]), "kg")

    cdadt_problem = om.Problem()
    cdadt_problem.model.add_subsystem(
        "aircraft",
        DisciplineGroup(
            disciplines=aircraft,
            config=b738_config,
            num_nodes=1,
            flight_phase="aircraft",
            apply_configured_values=True,
        ),
        promotes=["*"],
    )
    cdadt_problem.setup(check=False)

    reference_problem = om.Problem()
    reference_problem.model.add_subsystem("aircraft", _OpenConceptAircraftLevel(), promotes=["*"])
    reference_problem.setup(check=False)

    for problem in (cdadt_problem, reference_problem):
        for name, (value, units) in inputs.items():
            problem.set_val(name, value, units=units)
        problem.run_model()

    return cdadt_problem, reference_problem


@pytest.mark.validation
@pytest.mark.parametrize(("name", "units"), COMPARED_QUANTITIES, ids=[n for n, _ in COMPARED_QUANTITIES])
def test_aircraft_level_matches_openconcepts_assembly(aircraft_level_comparison, name, units):
    """cdadt's aircraft-level result equals OpenConcept's, evaluated at the same point.

    The tolerance is machine-level rather than engineering-level on purpose. cdadt's
    providers wrap these very components and add no physics, so anything but an exact match
    means cdadt has changed a number somewhere -- a unit, a wiring, a moment arm.
    """
    cdadt_problem, reference_problem = aircraft_level_comparison
    assert cdadt_problem.get_val(name, units=units) == pytest.approx(
        reference_problem.get_val(name, units=units), rel=1e-12
    )


@pytest.mark.validation
def test_the_comparison_is_not_vacuous(aircraft_level_comparison):
    """The compared quantities are real numbers, not zeros or defaults on both sides.

    Without this, a comparison in which both assemblies silently produced their placeholder
    defaults would pass and prove nothing.
    """
    cdadt_problem, _ = aircraft_level_comparison
    for name, units in COMPARED_QUANTITIES:
        value = cdadt_problem.get_val(name, units=units).item()
        assert np.isfinite(value)
        assert value > 0.0, f"{name} is {value}, which is not a physically meaningful result"

    # A B738-class airframe: sanity bounds wide enough to admit any reasonable design and
    # narrow enough to catch a units error or a dropped connection.
    assert 30e3 < cdadt_problem.get_val("ac|weights|OEW", units="kg").item() < 55e3
    assert 3.0 < cdadt_problem.get_val("ac|geom|wing|MAC", units="m").item() < 6.0


# ==============================================================================
# The converged sizing result
# ==============================================================================
@pytest.fixture(scope="module")
def sized_b738(b738_config):
    """Converge the B738 sizing loop and return the loop and its results."""
    loop = SizingLoop(
        jet_transport_disciplines(b738_config, engine_deck=ENGINE_DECK),
        b738_config,
        num_nodes=11,
    )
    problem = loop.build()
    loop.converge(problem)
    return loop, problem, loop.results(problem)


@pytest.mark.integration
def test_the_sizing_loop_converges_and_closes_the_weight_balance(sized_b738):
    """MTOW equals OEW plus payload plus total fuel, which is what the loop solves for."""
    _, _, results = sized_b738
    closure = results["OEW"] + results["payload"] + results["total_fuel"]
    assert results["MTOW"] == pytest.approx(closure, rel=1e-8), (
        f"MTOW {results['MTOW']:.1f} kg does not equal OEW + payload + fuel = {closure:.1f} kg; "
        f"the sizing loop reported a converged result that does not satisfy its own balance."
    )


@pytest.mark.integration
def test_the_sized_aircraft_flies_the_requested_mission(sized_b738):
    """Range flown equals range requested, and reserves are flown on top of it."""
    loop, _, results = sized_b738
    requested_range, units = loop.blackbox.profile.parameters["mission_range"]
    reserve_range, _ = loop.blackbox.profile.parameters["reserve_range"]
    assert units == "NM"

    assert results["mission_range_flown"] == pytest.approx(requested_range, rel=1e-4)
    assert results["reserve_range_flown"] == pytest.approx(requested_range + reserve_range, rel=1e-4)
    assert results["total_fuel"] > results["block_fuel"]


@pytest.mark.integration
def test_the_balanced_field_length_is_balanced(sized_b738):
    """Continue and abort distances match, which is what V1 is solved to achieve."""
    _, _, results = sized_b738
    assert results["takeoff_field_length"] > 0.0
    assert results["accelerate_stop_distance"] == pytest.approx(results["takeoff_field_length"], rel=1e-4)


@pytest.mark.integration
def test_takeoff_speeds_are_ordered(sized_b738):
    """Stall speed <= V1 <= V_R, as the takeoff decision requires."""
    _, _, results = sized_b738
    assert results["stall_speed_takeoff"] <= results["decision_speed"] + 1e-6
    assert results["decision_speed"] <= results["rotation_speed"] + 1e-6


@pytest.mark.integration
def test_the_engine_out_climb_gradient_is_positive(sized_b738):
    """The aircraft climbs on one engine, without which §25.121 cannot be satisfied."""
    _, _, results = sized_b738
    assert results["engine_out_climb_gradient"] > 0.0


@pytest.mark.regression
def test_the_converged_result_has_not_drifted(sized_b738):
    """Pin the converged B738 sizing result.

    This is a *regression* claim and nothing more: it says the answer has not changed, not
    that it is right. What establishes correctness is ``test_against_openconcept.py``, which
    compares these same quantities against OpenConcept's published golden values and against
    a live run of its example. This test exists to catch drift between those runs, cheaply.

    Note the values here are for the *shipped* configuration, which starts the ground roll
    from a configured speed rather than OpenConcept's 2 m/s. The validation tests override
    that back to match the reference exactly; this one does not, because its job is to pin
    what cdadt actually ships.

    Tolerances are loose enough to absorb platform floating-point differences and tight
    enough that any real modeling change trips them -- which is what they are for. These
    values moved once already, when the ground roll was changed to start from a configured
    speed rather than OpenConcept's 2 m/s stand-in for zero: the balanced field length
    shortened from 5247.7 ft to 5230.0 ft, because the aircraft is now credited with the
    first few metres of acceleration. That is a real modeling change with a real
    consequence, and the golden moving is the test doing its job.
    """
    _, _, results = sized_b738
    golden = {
        "MTOW": 78341.2,
        "OEW": 41747.4,
        "MLW": 62672.9,
        "block_fuel": 15973.5,
        "total_fuel": 18593.8,
        "takeoff_field_length": 5230.0,
        "decision_speed": 135.11,
        "engine_out_climb_gradient": 0.0422,
    }
    drifted = {
        name: (expected, results[name])
        for name, expected in golden.items()
        if results[name] != pytest.approx(expected, rel=1e-3)
    }
    assert not drifted, "Converged B738 results have drifted (expected, actual): " + repr(drifted)
