"""Tests for :mod:`cdadt.mission.aircraft_model`.

Claim class: ``contract`` and ``unit``. The factory exists because OpenConcept constructs the
aircraft model itself, with a fixed signature and no argument for configuration. These tests
prove the class it produces satisfies that signature, that two factories share no state, and
that a discipline set which cannot satisfy the contract is rejected at construction rather
than producing a model with unconnected inputs.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt.core.discipline import CouplingError
from cdadt.core.variables import Variable, VariableSet
from cdadt.mission.aircraft_model import AircraftModelFactory, CdadtAircraftModel
from cdadt.mission.contract import DRAG, MISSION_PHASES, THRUST, WEIGHT
from cdadt.tests.mission.support import (
    SimpleDragPolarProvider,
    SimpleFuelWeightProvider,
    SimpleThrustProvider,
    _Named,
    simple_config,
    simple_disciplines,
)


# ==============================================================================
# The produced class satisfies OpenConcept's construction signature
# ==============================================================================
@pytest.mark.contract
@pytest.mark.parametrize("spec", MISSION_PHASES, ids=lambda spec: spec.name)
def test_produced_class_is_constructible_the_way_openconcept_constructs_it(spec):
    """The class accepts exactly ``(num_nodes=..., flight_phase=...)``, for every phase.

    This is the call OpenConcept makes, verbatim, at five sites in
    ``openconcept/mission/phases.py``. A signature mismatch would surface as a ``TypeError``
    from deep inside OpenConcept's setup.
    """
    model_class = AircraftModelFactory(simple_disciplines()).build()
    instance = model_class(num_nodes=5, flight_phase=spec.name)

    assert isinstance(instance, om.Group)
    assert instance.options["num_nodes"] == 5
    assert instance.options["flight_phase"] == spec.name


@pytest.mark.unit
def test_produced_class_is_a_real_group_subclass():
    """The result is a class, not a partial or a closure, so it inspects like any group."""
    model_class = AircraftModelFactory(simple_disciplines()).build()
    assert isinstance(model_class, type)
    assert issubclass(model_class, CdadtAircraftModel)
    assert issubclass(model_class, om.Group)


@pytest.mark.unit
def test_produced_class_documents_its_disciplines():
    """The generated docstring names the disciplines, so an N2 or a repr is self-explaining."""
    model_class = AircraftModelFactory(simple_disciplines()).build()
    for name in ("aerodynamics", "propulsion", "weights"):
        assert name in model_class.__doc__


# ==============================================================================
# No shared state between models
# ==============================================================================
@pytest.mark.unit
def test_two_factories_produce_distinct_classes_with_distinct_disciplines():
    """Building a second aircraft does not disturb the first.

    Every sizing iteration and every optimizer function evaluation builds aircraft models.
    If the disciplines lived in a module-level registry, results would depend on the order
    models were built. Binding them to the class is what removes that coupling.
    """
    config = simple_config()
    first_disciplines = simple_disciplines(config)
    second_disciplines = [
        _Named(SimpleDragPolarProvider(config), config, "aerodynamics"),
        _Named(SimpleThrustProvider(config), config, "propulsion"),
        _Named(SimpleFuelWeightProvider(config), config, "weights"),
    ]

    first_class = AircraftModelFactory(first_disciplines).build()
    second_class = AircraftModelFactory(second_disciplines).build()

    assert first_class is not second_class
    assert first_class._cdadt_disciplines == tuple(first_disciplines)
    assert second_class._cdadt_disciplines == tuple(second_disciplines)
    assert first_class._cdadt_disciplines is not second_class._cdadt_disciplines


@pytest.mark.unit
def test_the_base_class_is_never_mutated():
    """Building models leaves :class:`CdadtAircraftModel` itself with no disciplines bound."""
    AircraftModelFactory(simple_disciplines()).build()
    assert CdadtAircraftModel._cdadt_disciplines == ()


# ==============================================================================
# Validation at construction
# ==============================================================================
@pytest.mark.unit
def test_a_discipline_set_that_produces_no_thrust_is_rejected():
    """Omitting a required output is caught when the factory is built.

    OpenConcept promotes the aircraft model with ``promotes_inputs=["*"]``, so a missing
    ``thrust`` does not raise there: the acceleration balance simply reads an unconnected
    input at its default and the mission converges to a number computed from it.
    """
    config = simple_config()
    aero_only = [_Named(SimpleDragPolarProvider(config), config, "aerodynamics")]

    with pytest.raises(ValueError) as excinfo:
        AircraftModelFactory(aero_only)

    message = str(excinfo.value)
    assert "'thrust'" in message
    assert "'weight'" in message


@pytest.mark.unit
def test_a_discipline_requiring_an_unavailable_flight_condition_is_rejected():
    """Consuming a variable OpenConcept does not supply is rejected, naming the phase.

    ``fltcond|singamma`` exists in the integrated phases but not in the rotation phase,
    which uses Raymer's circular-arc transition instead of an ODE. A provider consuming it
    would silently get a default of zero during rotation.
    """
    config = simple_config()

    class NeedsSinGamma(SimpleDragPolarProvider):
        def requires(self):
            return super().requires() | VariableSet([Variable("fltcond|singamma", None, vectorized=True)])

    disciplines = [
        _Named(NeedsSinGamma(config), config, "aerodynamics"),
        _Named(SimpleThrustProvider(config), config, "propulsion"),
        _Named(SimpleFuelWeightProvider(config), config, "weights"),
    ]

    with pytest.raises(CouplingError) as excinfo:
        AircraftModelFactory(disciplines)

    message = str(excinfo.value)
    assert "rotate" in message
    assert "fltcond|singamma" in message


@pytest.mark.unit
def test_two_disciplines_providing_the_same_variable_are_rejected():
    """A doubly-provided output would make the result depend on subsystem order."""
    config = simple_config()
    disciplines = [
        _Named(SimpleDragPolarProvider(config), config, "aerodynamics"),
        _Named(SimpleDragPolarProvider(config), config, "aerostructural"),
        _Named(SimpleThrustProvider(config), config, "propulsion"),
        _Named(SimpleFuelWeightProvider(config), config, "weights"),
    ]

    with pytest.raises(CouplingError, match="'drag' is provided by more than one discipline"):
        AircraftModelFactory(disciplines)


@pytest.mark.unit
def test_factory_repr_lists_the_disciplines():
    """The representation is informative in a failing assertion."""
    factory = AircraftModelFactory(simple_disciplines())
    assert repr(factory) == "AircraftModelFactory(['aerodynamics', 'propulsion', 'weights'])"


# ==============================================================================
# The built model actually computes
# ==============================================================================
@pytest.mark.integration
def test_the_built_model_produces_thrust_drag_and_mass():
    """Standing alone, the model turns flight conditions into the three required outputs.

    Values are checked against the closed form of the fixture providers, so a wiring error
    that swapped or dropped a connection changes the result rather than merely the shape.
    """
    num_nodes = 3
    model_class = AircraftModelFactory(simple_disciplines()).build()

    prob = om.Problem()
    prob.model.add_subsystem("acmodel", model_class(num_nodes=num_nodes, flight_phase="cruise"), promotes=["*"])
    prob.setup(check=False)

    prob.set_val("fltcond|CL", np.full(num_nodes, 0.5))
    prob.set_val("fltcond|q", np.full(num_nodes, 1.0e4), units="N * m**-2")
    prob.set_val("ac|geom|wing|S_ref", 124.6, units="m**2")
    prob.set_val("throttle", np.full(num_nodes, 0.8))
    prob.set_val("propulsor_active", np.ones(num_nodes))
    prob.set_val("fltcond|h", np.full(num_nodes, 10_000.0), units="m")
    prob.set_val("ac|weights|MTOW", 79e3, units="kg")
    # Standing alone, the integrator's duration is not promoted to a phase-level "duration".
    # Inside a mission, OpenConcept's PhaseGroup.configure() walks the aircraft model, finds
    # every Integrator, and connects the phase duration to it automatically -- which is why
    # the weights provider does not wire duration itself.
    prob.set_val("fuel_burn_integ.duration", 600.0, units="s")
    prob.run_model()

    # drag = (CD0 + k CL^2) q S = (0.02 + 0.05 * 0.25) * 1e4 * 124.6
    assert prob.get_val("drag", units="N") == pytest.approx(np.full(num_nodes, 0.0325 * 1.0e4 * 124.6))
    # thrust = T_sl (1 - lapse h) throttle = 220e3 * (1 - 5e-5 * 1e4) * 0.8
    assert prob.get_val("thrust", units="N") == pytest.approx(np.full(num_nodes, 220e3 * 0.5 * 0.8))
    # Mass starts at MTOW and decreases as fuel burns.
    mass = prob.get_val("weight", units="kg")
    assert mass[0] == pytest.approx(79e3)
    assert np.all(np.diff(mass) < 0.0)


@pytest.mark.integration
def test_a_failed_propulsor_halves_the_thrust():
    """``propulsor_active`` reaches the propulsion discipline and changes the result.

    OpenConcept sets this to zero in the engine-out phases. If it did not reach the model,
    the balanced field length and the 25.121 climb gradient would both be computed with all
    engines running.
    """
    num_nodes = 3
    model_class = AircraftModelFactory(simple_disciplines()).build()

    prob = om.Problem()
    prob.model.add_subsystem("acmodel", model_class(num_nodes=num_nodes, flight_phase="v1vr"), promotes=["*"])
    prob.setup(check=False)
    prob.set_val("throttle", np.ones(num_nodes))
    prob.set_val("fltcond|h", np.zeros(num_nodes), units="m")

    prob.set_val("propulsor_active", np.ones(num_nodes))
    prob.run_model()
    all_engines = prob.get_val("thrust", units="N").copy()

    prob.set_val("propulsor_active", np.zeros(num_nodes))
    prob.run_model()
    engine_out = prob.get_val("thrust", units="N")

    assert engine_out == pytest.approx(0.5 * all_engines)


@pytest.mark.unit
def test_required_outputs_declaration_matches_the_contract_constants():
    """The three contract variables are the ones the factory checks for.

    Guards against the declaration and the check drifting apart, which would let a model
    missing one of them through.
    """
    from cdadt.mission.contract import AIRCRAFT_MODEL_OUTPUTS

    assert set(AIRCRAFT_MODEL_OUTPUTS) == {THRUST, DRAG, WEIGHT}
