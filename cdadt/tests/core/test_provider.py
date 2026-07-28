"""Tests for :mod:`cdadt.core.provider`.

Claim class: ``unit``. These prove the provider contract: a provider must declare a name, a
citation, what it consumes and produces, and must build real OpenMDAO subsystems that
actually create what it declared.

The concrete providers defined here are not test doubles for a dependency. They are minimal
real providers built from real OpenMDAO components, used to exercise the abstract base
class itself. Every assertion about what a provider produces is made by building an
OpenMDAO model and running it.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt.core.configuration import AircraftConfiguration, MissingConfigurationError
from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

pytestmark = pytest.mark.unit


class ConstantDragProvider(Provider):
    """A minimal real provider: drag from a configured drag-area, D = k * q.

    Deliberately trivial physics. Its purpose is to be a genuine ``Provider`` whose output
    can be checked numerically, so the tests below verify the base-class contract against
    something that actually runs rather than against a stand-in.
    """

    @property
    def name(self) -> str:
        return "constant_drag_area"

    @property
    def reference(self) -> str:
        return "Test fixture; D = f * q with f a configured equivalent flat-plate area."

    def provides(self) -> VariableSet:
        return VariableSet([Variable("drag", "N", vectorized=True, description="Total drag force")])

    def requires(self) -> VariableSet:
        return VariableSet([Variable("fltcond|q", "N*m**-2", vectorized=True, description="Dynamic pressure")])

    def validate_configuration(self) -> None:
        self.config.require_all(["ac|aero|flat_plate_area"])

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        flat_plate_area = self.config.scalar("ac|aero|flat_plate_area", units="m**2")
        group.add_subsystem(
            "drag_calc",
            om.ExecComp(
                "drag = f * q",
                drag={"val": np.ones(num_nodes), "units": "N"},
                q={"val": np.ones(num_nodes), "units": "N*m**-2"},
                f={"val": flat_plate_area, "units": "m**2"},
                has_diag_partials=True,
            ),
            promotes_inputs=[("q", "fltcond|q")],
            promotes_outputs=["drag"],
        )


@pytest.fixture
def config():
    """Return a configuration containing the one constant the test provider needs."""
    return AircraftConfiguration({"ac": {"aero": {"flat_plate_area": {"value": 2.5, "units": "m**2"}}}})


def test_provider_cannot_be_instantiated_directly(config):
    """Provider is abstract; every subclass must supply the full interface."""
    with pytest.raises(TypeError):
        Provider(config)


def test_incomplete_subclass_cannot_be_instantiated(config):
    """Omitting any abstract member keeps the subclass abstract."""

    class MissingBuild(Provider):
        @property
        def name(self):
            return "incomplete"

        @property
        def reference(self):
            return "none"

        def provides(self):
            return VariableSet()

        def requires(self):
            return VariableSet()

    with pytest.raises(TypeError):
        MissingBuild(config)


def test_provider_exposes_name_reference_and_config(config):
    """A provider reports its identity, its citation, and the configuration it reads."""
    provider = ConstantDragProvider(config)
    assert provider.name == "constant_drag_area"
    assert "flat-plate area" in provider.reference
    assert provider.config is config


def test_validate_configuration_runs_at_construction():
    """A provider missing a required constant fails when built, not during a solve.

    This is the point of the no-defaults rule: the failure happens once, at construction,
    with the name of the absent entry, rather than as a converged answer computed from a
    fallback value.
    """
    empty = AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 1.0, "units": "m**2"}}}}})
    with pytest.raises(MissingConfigurationError, match="ac\\|aero\\|flat_plate_area"):
        ConstantDragProvider(empty)


def test_declared_outputs_are_actually_created_and_correct(config):
    """Building the provider produces the promoted output it declared, with the right value.

    ``provides()`` is a promise. This checks the promise is kept by building a real model
    and running it, rather than by inspecting the declaration.
    """
    num_nodes = 5
    provider = ConstantDragProvider(config)

    prob = om.Problem()
    group = prob.model.add_subsystem("acmodel", om.Group(), promotes=["*"])
    provider.build(group, num_nodes=num_nodes, flight_phase="cruise")
    prob.setup(check=False)

    dynamic_pressure = np.linspace(1.0e4, 1.5e4, num_nodes)
    prob.set_val("fltcond|q", dynamic_pressure, units="N*m**-2")
    prob.run_model()

    assert prob.get_val("drag", units="N") == pytest.approx(2.5 * dynamic_pressure)


def test_declared_inputs_match_the_built_model(config):
    """Every variable in ``requires()`` appears as a promoted input of the built group."""
    provider = ConstantDragProvider(config)

    prob = om.Problem()
    group = prob.model.add_subsystem("acmodel", om.Group(), promotes=["*"])
    provider.build(group, num_nodes=3, flight_phase="cruise")
    prob.setup(check=False)

    promoted_inputs = {
        meta["prom_name"] for meta in prob.model.get_io_metadata("input", metadata_keys=["units"]).values()
    }
    for variable in provider.requires():
        assert variable.name in promoted_inputs


def test_declared_units_match_the_built_model(config):
    """Declared units agree with the units OpenMDAO assigned, so conversions are honest."""
    provider = ConstantDragProvider(config)

    prob = om.Problem()
    group = prob.model.add_subsystem("acmodel", om.Group(), promotes=["*"])
    provider.build(group, num_nodes=3, flight_phase="cruise")
    prob.setup(check=False)

    metadata = prob.model.get_io_metadata("output", metadata_keys=["units"], return_rel_names=False)
    units_by_prom = {meta["prom_name"]: meta["units"] for meta in metadata.values()}
    for variable in provider.provides():
        assert units_by_prom[variable.name] == variable.units


def test_provider_repr_names_the_class_and_identifier(config):
    """The representation is informative in a failing assertion."""
    assert repr(ConstantDragProvider(config)) == "ConstantDragProvider(name='constant_drag_area')"


def test_base_validate_configuration_accepts_any_configuration(config):
    """A provider with no method constants need not override validate_configuration."""

    class NoConstants(Provider):
        @property
        def name(self):
            return "no_constants"

        @property
        def reference(self):
            return "none"

        def provides(self):
            return VariableSet()

        def requires(self):
            return VariableSet()

        def build(self, group, num_nodes, flight_phase):
            group.add_subsystem("noop", om.ExecComp("y = x"))

    assert NoConstants(AircraftConfiguration({"ac": {"x": {"value": 1.0}}})).name == "no_constants"
