"""Tests for the OpenConcept-backed providers.

Claim classes: ``unit`` for declaration and configuration behavior, ``validation`` for the
equivalence checks.

The central question about a wrapper is whether it is faithful: does running the physics
through cdadt give the same answer as running the wrapped component directly? Each
equivalence test below builds the raw OpenConcept component in one problem and the cdadt
provider in another, feeds both identical inputs, and asserts the outputs match exactly.
That is a stronger claim than "the provider produces a plausible number", and it is the
claim that matters, because the point of a provider is to add no physics of its own.

Nothing here is mocked. Every test runs the real OpenConcept components.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest
from openconcept.aerodynamics import CleanCLmax, FlapCLmax, ParasiteDragCoefficient_JetTransport, PolarDrag
from openconcept.geometry import CylinderSurfaceArea, WingMACTrapezoidal
from openconcept.propulsion import RubberizedTurbofan
from openconcept.stability import HStabVolumeCoefficientSizing, VStabVolumeCoefficientSizing

from cdadt.core.configuration import AircraftConfiguration, MissingConfigurationError
from cdadt.core.discipline import Discipline, DisciplineGroup, DisciplineScope
from cdadt.disciplines import (
    Aerodynamics,
    EmptyWeight,
    Geometry,
    MassBookkeeping,
    MaximumLift,
    Propulsion,
    Stability,
)
from cdadt.providers.openconcept import (
    FuelBurnMassProvider,
    JetTransportDragProvider,
    JetTransportEmptyWeightProvider,
    JetTransportMaximumLiftProvider,
    RubberizedTurbofanProvider,
    TailVolumeCoefficientProvider,
    TrapezoidalGeometryProvider,
)

NUM_NODES = 3


class _SingleProviderDiscipline(Discipline):
    """A discipline wrapping exactly one provider, so it can be built on its own."""

    @property
    def name(self) -> str:
        return "provider_under_test"


def _run_provider(provider, inputs, flight_phase="cruise", num_nodes=1, config=None):
    """Build one provider into a problem, set inputs, run, and return the problem.

    The provider is built through a real :class:`~cdadt.core.discipline.DisciplineGroup`,
    the same machinery the assembled model uses, so input reconciliation behaves here
    exactly as it does in a mission rather than being reproduced by the test.

    Parameters
    ----------
    provider : Provider
        Provider to build.
    inputs : dict
        Mapping of promoted name to ``(value, units)``.
    flight_phase : str, optional
        Phase to build for.
    num_nodes : int, optional
        Node count to build for.
    config : AircraftConfiguration, optional
        Configuration used to reconcile input defaults.

    Returns
    -------
    openmdao.api.Problem
        The run problem.
    """
    discipline = _SingleProviderDiscipline(provider, config or provider.config)
    prob = om.Problem()
    prob.model.add_subsystem(
        "under_test",
        DisciplineGroup(
            disciplines=[discipline],
            config=config,
            num_nodes=num_nodes,
            flight_phase=flight_phase,
            # Standing alone, this group is the highest point at which its inputs are
            # promoted, so seeding configured values here is correct. Inside a mission it
            # would not be; see DisciplineGroup.
            apply_configured_values=config is not None,
        ),
        promotes=["*"],
    )

    # Declare the inputs above the group, before setup. Built in isolation a provider has no
    # sibling disciplines feeding it, and where two wrapped OpenConcept components declare
    # different placeholder values for the same promoted input, OpenMDAO refuses to pick one
    # -- correctly. Inside an assembled model those inputs have a source and the question
    # does not arise.
    for name, (value, units) in inputs.items():
        prob.model.set_input_defaults(name, val=value, units=units)

    prob.setup(check=False)
    for name, (value, units) in inputs.items():
        prob.set_val(name, value, units=units)
    prob.run_model()
    return prob


def _run_component(component, inputs):
    """Build one bare OpenConcept component into a problem, run it, and return the problem."""
    prob = om.Problem()
    prob.model.add_subsystem("comp", component, promotes=["*"])
    prob.setup(check=False)
    for name, (value, units) in inputs.items():
        prob.set_val(name, value, units=units)
    prob.run_model()
    return prob


# ==============================================================================
# Geometry
# ==============================================================================
@pytest.mark.validation
def test_wing_mac_matches_the_wrapped_component(b738_config):
    """The provider's MAC is exactly OpenConcept's WingMACTrapezoidal result."""
    geometry_inputs = {
        "ac|geom|wing|S_ref": (124.6, "m**2"),
        "ac|geom|wing|AR": (9.45, None),
        "ac|geom|wing|taper": (0.159, None),
        "ac|geom|fuselage|length": (38.08, "m"),
        "ac|geom|fuselage|height": (3.76, "m"),
        "ac|geom|nacelle|length": (4.3, "m"),
        "ac|geom|nacelle|diameter": (2.0, "m"),
    }
    through_provider = _run_provider(TrapezoidalGeometryProvider(b738_config), geometry_inputs, config=b738_config)
    direct = _run_component(
        WingMACTrapezoidal(),
        {"S_ref": (124.6, "m**2"), "AR": (9.45, None), "taper": (0.159, None)},
    )
    assert through_provider.get_val("ac|geom|wing|MAC", units="m") == pytest.approx(
        direct.get_val("MAC", units="m"), rel=1e-14
    )


@pytest.mark.validation
def test_wetted_areas_match_the_wrapped_component(b738_config):
    """Fuselage and nacelle wetted areas are exactly CylinderSurfaceArea results."""
    geometry_inputs = {
        "ac|geom|wing|S_ref": (124.6, "m**2"),
        "ac|geom|wing|AR": (9.45, None),
        "ac|geom|wing|taper": (0.159, None),
        "ac|geom|fuselage|length": (38.08, "m"),
        "ac|geom|fuselage|height": (3.76, "m"),
        "ac|geom|nacelle|length": (4.3, "m"),
        "ac|geom|nacelle|diameter": (2.0, "m"),
    }
    through_provider = _run_provider(TrapezoidalGeometryProvider(b738_config), geometry_inputs, config=b738_config)

    fuselage = _run_component(CylinderSurfaceArea(), {"L": (38.08, "m"), "D": (3.76, "m")})
    nacelle = _run_component(CylinderSurfaceArea(), {"L": (4.3, "m"), "D": (2.0, "m")})

    assert through_provider.get_val("ac|geom|fuselage|S_wet", units="m**2") == pytest.approx(
        fuselage.get_val("A", units="m**2"), rel=1e-14
    )
    assert through_provider.get_val("ac|geom|nacelle|S_wet", units="m**2") == pytest.approx(
        nacelle.get_val("A", units="m**2"), rel=1e-14
    )


@pytest.mark.unit
def test_tail_moment_arms_use_the_configured_fractions(b738_config):
    """Each tail arm is its configured fraction of fuselage length, not a built-in 0.5."""
    config = b738_config.with_overrides(
        {
            "ac|geom|hstab|arm_fraction_of_fuselage": 0.45,
            "ac|geom|vstab|arm_fraction_of_fuselage": 0.55,
        }
    )
    geometry_inputs = {
        "ac|geom|wing|S_ref": (124.6, "m**2"),
        "ac|geom|wing|AR": (9.45, None),
        "ac|geom|wing|taper": (0.159, None),
        "ac|geom|fuselage|length": (40.0, "m"),
        "ac|geom|fuselage|height": (3.76, "m"),
        "ac|geom|nacelle|length": (4.3, "m"),
        "ac|geom|nacelle|diameter": (2.0, "m"),
    }
    prob = _run_provider(TrapezoidalGeometryProvider(config), geometry_inputs, config=config)

    assert prob.get_val("ac|geom|hstab|c4_to_wing_c4", units="m").item() == pytest.approx(18.0)
    assert prob.get_val("ac|geom|vstab|c4_to_wing_c4", units="m").item() == pytest.approx(22.0)


@pytest.mark.unit
def test_geometry_provider_requires_its_configured_constants(b738_config):
    """Omitting a tail-arm fraction fails at construction, naming what is absent."""
    without_fractions = AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}})
    with pytest.raises(MissingConfigurationError) as excinfo:
        TrapezoidalGeometryProvider(without_fractions)
    message = str(excinfo.value)
    assert "ac|geom|hstab|arm_fraction_of_fuselage" in message
    assert "ac|geom|vstab|arm_fraction_of_fuselage" in message


# ==============================================================================
# Stability
# ==============================================================================
@pytest.mark.validation
def test_tail_areas_match_the_wrapped_components(b738_config):
    """Stabilizer areas are exactly OpenConcept's volume-coefficient sizing results."""
    stability_inputs = {
        "ac|geom|wing|S_ref": (124.6, "m**2"),
        "ac|geom|wing|AR": (9.45, None),
        "ac|geom|wing|MAC": (4.2684, "m"),
        "ac|geom|hstab|c4_to_wing_c4": (19.04, "m"),
        "ac|geom|vstab|c4_to_wing_c4": (19.04, "m"),
    }
    through_provider = _run_provider(TailVolumeCoefficientProvider(b738_config), stability_inputs, config=b738_config)

    hstab = _run_component(
        HStabVolumeCoefficientSizing(C_ht=b738_config.scalar("ac|geom|hstab|volume_coefficient")),
        {
            "ac|geom|wing|S_ref": (124.6, "m**2"),
            "ac|geom|wing|MAC": (4.2684, "m"),
            "ac|geom|hstab|c4_to_wing_c4": (19.04, "m"),
        },
    )
    vstab = _run_component(
        VStabVolumeCoefficientSizing(C_vt=b738_config.scalar("ac|geom|vstab|volume_coefficient")),
        {
            "ac|geom|wing|S_ref": (124.6, "m**2"),
            "ac|geom|wing|AR": (9.45, None),
            "ac|geom|vstab|c4_to_wing_c4": (19.04, "m"),
        },
    )

    assert through_provider.get_val("ac|geom|hstab|S_ref", units="m**2") == pytest.approx(
        hstab.get_val("ac|geom|hstab|S_ref", units="m**2"), rel=1e-12
    )
    assert through_provider.get_val("ac|geom|vstab|S_ref", units="m**2") == pytest.approx(
        vstab.get_val("ac|geom|vstab|S_ref", units="m**2"), rel=1e-12
    )


@pytest.mark.unit
def test_tail_volume_coefficients_come_from_configuration_not_openconcept_defaults(b738_config):
    """Changing the configured coefficient changes the tail area proportionally.

    OpenConcept declares these as options with Raymer jet-transport defaults. This asserts
    cdadt reads the configuration instead: a provider that silently inherited the option
    default would give the same area for both configurations below.
    """
    stability_inputs = {
        "ac|geom|wing|S_ref": (124.6, "m**2"),
        "ac|geom|wing|AR": (9.45, None),
        "ac|geom|wing|MAC": (4.2684, "m"),
        "ac|geom|hstab|c4_to_wing_c4": (19.04, "m"),
        "ac|geom|vstab|c4_to_wing_c4": (19.04, "m"),
    }
    baseline = _run_provider(TailVolumeCoefficientProvider(b738_config), stability_inputs, config=b738_config)

    doubled_config = b738_config.with_overrides(
        {"ac|geom|hstab|volume_coefficient": 2.0 * b738_config.scalar("ac|geom|hstab|volume_coefficient")}
    )
    doubled = _run_provider(TailVolumeCoefficientProvider(doubled_config), stability_inputs, config=doubled_config)

    assert doubled.get_val("ac|geom|hstab|S_ref", units="m**2") == pytest.approx(
        2.0 * baseline.get_val("ac|geom|hstab|S_ref", units="m**2"), rel=1e-12
    )


@pytest.mark.unit
def test_stability_provider_requires_its_volume_coefficients():
    """Omitting a volume coefficient fails at construction rather than using Raymer's default."""
    bare = AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}})
    with pytest.raises(MissingConfigurationError) as excinfo:
        TailVolumeCoefficientProvider(bare)
    assert "ac|geom|hstab|volume_coefficient" in str(excinfo.value)


# ==============================================================================
# Maximum lift
# ==============================================================================
@pytest.mark.validation
def test_maximum_lift_matches_the_wrapped_components(b738_config):
    """Clean and takeoff CLmax are exactly OpenConcept's CleanCLmax and FlapCLmax results."""
    clmax_inputs = {
        "ac|aero|airfoil_Cl_max": (1.75, None),
        "ac|geom|wing|c4sweep": (25.0, "deg"),
        "ac|geom|wing|toverc": (0.12, None),
        "ac|aero|takeoff_flap_deg": (15.0, "deg"),
    }
    through_provider = _run_provider(JetTransportMaximumLiftProvider(b738_config), clmax_inputs, config=b738_config)

    clean = _run_component(
        CleanCLmax(), {"ac|aero|airfoil_Cl_max": (1.75, None), "ac|geom|wing|c4sweep": (25.0, "deg")}
    )
    clean_clmax = clean.get_val("CL_max_clean").item()
    flapped = _run_component(
        FlapCLmax(),
        {
            "flap_extension": (15.0, "deg"),
            "ac|geom|wing|c4sweep": (25.0, "deg"),
            "ac|geom|wing|toverc": (0.12, None),
            "CL_max_clean": (clean_clmax, None),
        },
    )

    assert through_provider.get_val("ac|aero|CLmax_cruise") == pytest.approx(clean_clmax, rel=1e-14)
    assert through_provider.get_val("ac|aero|CLmax_TO") == pytest.approx(flapped.get_val("CL_max_flap"), rel=1e-14)


@pytest.mark.unit
def test_takeoff_clmax_exceeds_clean_clmax(b738_config):
    """Deploying flaps increases maximum lift, as it must for the takeoff speeds to make sense."""
    prob = _run_provider(
        JetTransportMaximumLiftProvider(b738_config),
        {
            "ac|aero|airfoil_Cl_max": (1.75, None),
            "ac|geom|wing|c4sweep": (25.0, "deg"),
            "ac|geom|wing|toverc": (0.12, None),
            "ac|aero|takeoff_flap_deg": (15.0, "deg"),
        },
        config=b738_config,
    )
    assert prob.get_val("ac|aero|CLmax_TO").item() > prob.get_val("ac|aero|CLmax_cruise").item()


# ==============================================================================
# Drag
# ==============================================================================
def _drag_inputs(num_nodes):
    """Return a full set of inputs for the drag buildup at a cruise condition."""
    return {
        "fltcond|Utrue": (np.full(num_nodes, 230.0), "m/s"),
        "fltcond|rho": (np.full(num_nodes, 0.38), "kg/m**3"),
        "fltcond|T": (np.full(num_nodes, 219.0), "degK"),
        "fltcond|q": (np.full(num_nodes, 1.0e4), "N/m**2"),
        "fltcond|CL": (np.full(num_nodes, 0.5), None),
        "ac|geom|fuselage|length": (38.08, "m"),
        "ac|geom|fuselage|height": (3.76, "m"),
        "ac|geom|fuselage|S_wet": (449.8, "m**2"),
        "ac|geom|hstab|S_ref": (27.9, "m**2"),
        "ac|geom|hstab|AR": (6.16, None),
        "ac|geom|hstab|taper": (0.203, None),
        "ac|geom|hstab|toverc": (0.12, None),
        "ac|geom|vstab|S_ref": (20.2, "m**2"),
        "ac|geom|vstab|AR": (1.91, None),
        "ac|geom|vstab|taper": (0.271, None),
        "ac|geom|vstab|toverc": (0.12, None),
        "ac|geom|wing|S_ref": (124.6, "m**2"),
        "ac|geom|wing|AR": (9.45, None),
        "ac|geom|wing|taper": (0.159, None),
        "ac|geom|wing|toverc": (0.12, None),
        "ac|geom|nacelle|length": (4.3, "m"),
        "ac|geom|nacelle|S_wet": (27.0, "m**2"),
        "ac|propulsion|num_engines": (2.0, None),
    }


@pytest.mark.validation
def test_cruise_drag_matches_the_wrapped_components(b738_config):
    """Clean-configuration drag is exactly the OpenConcept buildup fed into PolarDrag."""
    inputs = _drag_inputs(NUM_NODES)
    through_provider = _run_provider(
        JetTransportDragProvider(b738_config),
        inputs,
        flight_phase="cruise",
        num_nodes=NUM_NODES,
        config=b738_config,
    )

    buildup_inputs = {k: v for k, v in inputs.items() if not k.startswith(("fltcond|q", "fltcond|CL"))}
    buildup = _run_component(
        ParasiteDragCoefficient_JetTransport(num_nodes=NUM_NODES, configuration="clean"), buildup_inputs
    )
    cd0 = buildup.get_val("CD0")

    polar = _run_component(
        PolarDrag(num_nodes=NUM_NODES, vec_CD0=True),
        {
            "fltcond|CL": (np.full(NUM_NODES, 0.5), None),
            "fltcond|q": (np.full(NUM_NODES, 1.0e4), "N/m**2"),
            "ac|geom|wing|S_ref": (124.6, "m**2"),
            "ac|geom|wing|AR": (9.45, None),
            "CD0": (cd0, None),
            "e": (b738_config.scalar("ac|aero|polar|e"), None),
        },
    )

    assert through_provider.get_val("CD0") == pytest.approx(cd0, rel=1e-14)
    assert through_provider.get_val("drag", units="N") == pytest.approx(polar.get_val("drag", units="N"), rel=1e-14)


@pytest.mark.validation
@pytest.mark.parametrize("takeoff_phase", ["v0v1", "v1vr", "v1v0", "rotate"])
def test_takeoff_phases_use_the_takeoff_drag_configuration(b738_config, takeoff_phase):
    """Every balanced-field phase builds the buildup in takeoff configuration.

    The §25.113 field length and the §25.121 climb gradient are both written about an
    aircraft with flaps deployed. Running clean drag through these phases would understate
    both, and the mission would converge anyway.
    """
    inputs = _drag_inputs(NUM_NODES)
    inputs.update(
        {
            "ac|aero|takeoff_flap_deg": (15.0, "deg"),
            "ac|geom|wing|c4sweep": (25.0, "deg"),
        }
    )
    takeoff = _run_provider(
        JetTransportDragProvider(b738_config),
        inputs,
        flight_phase=takeoff_phase,
        num_nodes=NUM_NODES,
        config=b738_config,
    )

    expected = _run_component(
        ParasiteDragCoefficient_JetTransport(num_nodes=NUM_NODES, configuration="takeoff"),
        {k: v for k, v in inputs.items() if not k.startswith(("fltcond|q", "fltcond|CL"))},
    )
    assert takeoff.get_val("CD0") == pytest.approx(expected.get_val("CD0"), rel=1e-14)


@pytest.mark.validation
def test_takeoff_drag_exceeds_clean_drag(b738_config):
    """Flaps down produces more zero-lift drag than clean, at the same flight condition.

    An independent check on the configuration branch: if the phase test above were
    comparing two identical clean buildups, this would fail.

    The flap setting and sweep are supplied only to the takeoff build. In the clean
    configuration those inputs do not exist -- OpenConcept's buildup does not create them --
    which is itself evidence the branch is real.
    """
    clean_inputs = _drag_inputs(NUM_NODES)
    takeoff_inputs = {
        **clean_inputs,
        "ac|aero|takeoff_flap_deg": (15.0, "deg"),
        "ac|geom|wing|c4sweep": (25.0, "deg"),
    }

    clean = _run_provider(
        JetTransportDragProvider(b738_config), clean_inputs, "cruise", NUM_NODES, b738_config
    ).get_val("CD0")
    takeoff = _run_provider(
        JetTransportDragProvider(b738_config), takeoff_inputs, "v0v1", NUM_NODES, b738_config
    ).get_val("CD0")

    assert np.all(takeoff > clean), f"Takeoff CD0 {takeoff} is not greater than clean CD0 {clean}"


@pytest.mark.unit
def test_drag_provider_requires_the_oswald_efficiency():
    """Omitting the Oswald efficiency fails at construction."""
    bare = AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}})
    with pytest.raises(MissingConfigurationError, match=r"ac\|aero\|polar\|e"):
        JetTransportDragProvider(bare)


# ==============================================================================
# Propulsion
# ==============================================================================
@pytest.mark.validation
def test_thrust_matches_the_engine_deck_times_operating_engines(b738_config):
    """Installed thrust is the deck's per-engine thrust multiplied by the engine count."""
    inputs = {
        "throttle": (np.full(NUM_NODES, 0.9), None),
        "fltcond|h": (np.full(NUM_NODES, 9000.0), "m"),
        "fltcond|M": (np.full(NUM_NODES, 0.78), None),
        "propulsor_active": (np.ones(NUM_NODES), None),
        "ac|propulsion|engine|rating": (27000.0, "lbf"),
        "ac|propulsion|num_engines": (2.0, None),
    }
    through_provider = _run_provider(
        RubberizedTurbofanProvider(b738_config, deck="CFM56"),
        inputs,
        num_nodes=NUM_NODES,
        config=b738_config,
    )

    deck = _run_component(
        RubberizedTurbofan(num_nodes=NUM_NODES, engine="CFM56"),
        {
            "throttle": (np.full(NUM_NODES, 0.9), None),
            "fltcond|h": (np.full(NUM_NODES, 9000.0), "m"),
            "fltcond|M": (np.full(NUM_NODES, 0.78), None),
            "ac|propulsion|engine|rating": (27000.0, "lbf"),
        },
    )

    assert through_provider.get_val("thrust", units="lbf") == pytest.approx(
        2.0 * deck.get_val("thrust", units="lbf"), rel=1e-12
    )
    assert through_provider.get_val("fuel_flow", units="kg/s") == pytest.approx(
        2.0 * deck.get_val("fuel_flow", units="kg/s"), rel=1e-12
    )


@pytest.mark.validation
def test_a_failed_propulsor_removes_exactly_one_engine(b738_config):
    """With propulsor_active zero, a twin produces one engine's thrust, not two or none.

    OpenConcept sets propulsor_active to zero in the phases that establish the balanced
    field length and the §25.121 climb gradient. A provider that ignored it would size the
    aircraft against an all-engines-operating takeoff and converge just as happily.
    """
    inputs = {
        "throttle": (np.ones(NUM_NODES), None),
        "fltcond|h": (np.zeros(NUM_NODES), "m"),
        "fltcond|M": (np.full(NUM_NODES, 0.25), None),
        "propulsor_active": (np.ones(NUM_NODES), None),
        "ac|propulsion|engine|rating": (27000.0, "lbf"),
        "ac|propulsion|num_engines": (2.0, None),
    }
    provider = RubberizedTurbofanProvider(b738_config, deck="CFM56")
    all_engines = _run_provider(provider, inputs, num_nodes=NUM_NODES, config=b738_config).get_val(
        "thrust", units="lbf"
    )

    inputs["propulsor_active"] = (np.zeros(NUM_NODES), None)
    engine_out = _run_provider(provider, inputs, num_nodes=NUM_NODES, config=b738_config).get_val("thrust", units="lbf")

    assert engine_out == pytest.approx(0.5 * all_engines, rel=1e-12)


@pytest.mark.unit
def test_engine_deck_must_be_named_explicitly_and_validly(b738_config):
    """The deck is a required argument, and an unknown deck is rejected with the valid list."""
    with pytest.raises(TypeError):
        RubberizedTurbofanProvider(b738_config)

    with pytest.raises(ValueError, match="Valid decks"):
        RubberizedTurbofanProvider(b738_config, deck="CFM57")


@pytest.mark.unit
def test_engine_deck_is_reported_in_the_reference(b738_config):
    """The citation names the deck actually in use, so a run report is unambiguous."""
    assert "CFM56" in RubberizedTurbofanProvider(b738_config, deck="CFM56").reference
    assert "N3" in RubberizedTurbofanProvider(b738_config, deck="N3").reference


# ==============================================================================
# Weights
# ==============================================================================
@pytest.mark.unit
def test_landing_weight_uses_the_configured_fraction(b738_config):
    """Maximum landing mass is the configured fraction of MTOW, not a built-in 0.8."""
    config = b738_config.with_overrides({"ac|weights|MLW_fraction_of_MTOW": 0.85})
    # The tail areas are supplied explicitly. Built in isolation this provider has no
    # stability discipline feeding them, and two components inside OpenConcept's
    # empty-weight buildup declare different placeholder values for the same promoted
    # input, which OpenMDAO refuses to resolve on its own.
    prob = _run_provider(
        JetTransportEmptyWeightProvider(config),
        {
            "ac|weights|MTOW": (80000.0, "kg"),
            "ac|geom|hstab|S_ref": (27.9, "m**2"),
            "ac|geom|vstab|S_ref": (20.2, "m**2"),
        },
        config=config,
    )
    assert prob.get_val("ac|weights|MLW", units="kg").item() == pytest.approx(0.85 * 80000.0)


@pytest.mark.unit
def test_empty_weight_provider_requires_the_landing_weight_fraction():
    """Omitting the landing-weight fraction fails at construction."""
    bare = AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}})
    with pytest.raises(MissingConfigurationError, match=r"ac\|weights\|MLW_fraction_of_MTOW"):
        JetTransportEmptyWeightProvider(bare)


@pytest.mark.integration
def test_fuel_burn_mass_decreases_from_takeoff_mass(b738_config):
    """Mass starts at MTOW and decreases as fuel is integrated."""
    prob = _run_provider(
        FuelBurnMassProvider(b738_config),
        {
            "fuel_flow": (np.full(NUM_NODES, 1.0), "kg/s"),
            "ac|weights|MTOW": (79000.0, "kg"),
            "fuel_burn_integ.duration": (600.0, "s"),
        },
        num_nodes=NUM_NODES,
        config=b738_config,
    )
    mass = prob.get_val("weight", units="kg")
    assert mass[0] == pytest.approx(79000.0)
    assert np.all(np.diff(mass) < 0.0)
    # Constant 1 kg/s for 600 s burns 600 kg.
    assert prob.get_val("fuel_burn_integ.fuel_burn_final", units="kg").item() == pytest.approx(600.0, rel=1e-10)


@pytest.mark.unit
def test_the_integrator_is_named_as_the_mission_contract_expects(b738_config):
    """The fuel integrator's subsystem name matches the paths declared in the contract.

    :data:`cdadt.mission.contract.MISSION_OUTPUTS` reads total fuel from
    ``loiter.fuel_burn_integ.fuel_burn_final``. Renaming the integrator would break that
    path, and the mission would still converge.
    """
    from cdadt.mission.contract import MISSION_OUTPUTS_BY_NAME

    group = om.Group()
    FuelBurnMassProvider(b738_config).build(group, num_nodes=1, flight_phase="cruise")
    subsystem_names = {name for name, _ in group._static_subsystems_allprocs.items()}

    total_fuel_path = MISSION_OUTPUTS_BY_NAME["total_fuel"].path
    integrator_name = total_fuel_path.split(".")[1]
    assert integrator_name in subsystem_names, (
        f"The contract reads total fuel from '{total_fuel_path}', which requires a subsystem named "
        f"'{integrator_name}'. This provider builds: {sorted(subsystem_names)}"
    )


# ==============================================================================
# Discipline scopes
# ==============================================================================
@pytest.mark.unit
@pytest.mark.parametrize(
    ("discipline_class", "expected_scope"),
    [
        (Geometry, DisciplineScope.AIRCRAFT),
        (Stability, DisciplineScope.AIRCRAFT),
        (EmptyWeight, DisciplineScope.AIRCRAFT),
        (MaximumLift, DisciplineScope.AIRCRAFT),
        (Aerodynamics, DisciplineScope.PHASE),
        (Propulsion, DisciplineScope.PHASE),
        (MassBookkeeping, DisciplineScope.PHASE),
    ],
)
def test_discipline_scopes(discipline_class, expected_scope, b738_config):
    """Each discipline is scoped where its physics belongs.

    Getting this wrong is not cosmetic. An aircraft-scoped discipline built inside the
    phases would be evaluated twelve times, and its outputs would be invisible at the
    mission level where OpenConcept consumes ``ac|aero|CLmax_TO`` and ``ac|weights|MTOW``.
    """
    provider = TrapezoidalGeometryProvider(b738_config)
    assert discipline_class(provider, b738_config).scope is expected_scope
