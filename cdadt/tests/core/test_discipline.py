"""Tests for :mod:`cdadt.core.discipline`.

Claim class: ``unit``. These prove the discipline abstraction delegates correctly to its
provider and that :func:`~cdadt.core.discipline.check_coupling` refuses a discipline set
that cannot be connected.

The coupling check is the interesting one. Without it, a required input with no source
becomes an OpenMDAO input sitting quietly at its declared default, and the model converges
to an answer computed from a number nobody chose. The tests assert that both failure modes
-- an unsatisfied requirement and a doubly-provided variable -- are reported, and that every
problem in a set is reported at once rather than one per run.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
import pytest

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.discipline import CouplingError, Discipline, check_coupling
from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

pytestmark = pytest.mark.unit


class _ExecCompProvider(Provider):
    """A real provider built from an OpenMDAO ``ExecComp``, parameterized for these tests.

    Not a test double: it builds and runs actual OpenMDAO components. It is parameterized
    so the coupling scenarios below can be constructed without writing a class per case.
    """

    def __init__(self, config, identifier, expression, provided, required):
        self._identifier = identifier
        self._expression = expression
        self._provided = provided
        self._required = required
        super().__init__(config)

    @property
    def name(self):
        return self._identifier

    @property
    def reference(self):
        return f"Test fixture provider '{self._identifier}'"

    def provides(self):
        return self._provided

    def requires(self):
        return self._required

    def build(self, group, num_nodes, flight_phase):
        kwargs = {}
        for variable in [*self._provided, *self._required]:
            kwargs[variable.name.replace("|", "_")] = {
                "val": np.ones(num_nodes) if variable.vectorized else 1.0,
                "units": variable.units,
            }
        group.add_subsystem(
            f"{self._identifier}_comp",
            om.ExecComp(self._expression, has_diag_partials=True, **kwargs),
            promotes_inputs=[(v.name.replace("|", "_"), v.name) for v in self._required],
            promotes_outputs=[(v.name.replace("|", "_"), v.name) for v in self._provided],
        )


class _NamedDiscipline(Discipline):
    """A discipline whose only specialization is its name."""

    def __init__(self, provider, config, name):
        self._name = name
        super().__init__(provider, config)

    @property
    def name(self):
        return self._name


@pytest.fixture
def config():
    """Return a configuration with a single entry; these providers need no constants."""
    return AircraftConfiguration({"ac": {"geom": {"wing": {"S_ref": {"value": 124.6, "units": "m**2"}}}}})


def _make(config, identifier, expression, provided, required):
    """Build a named discipline wrapping an ``ExecComp`` provider."""
    provider = _ExecCompProvider(config, identifier, expression, VariableSet(provided), VariableSet(required))
    return _NamedDiscipline(provider, config, identifier)


DRAG = Variable("drag", "N", vectorized=True)
THRUST = Variable("thrust", "N", vectorized=True)
DYNAMIC_PRESSURE = Variable("fltcond|q", "N*m**-2", vectorized=True)
THROTTLE = Variable("throttle", None, vectorized=True)


# ==============================================================================
# The abstraction itself
# ==============================================================================
def test_discipline_cannot_be_instantiated_without_a_name(config):
    """Discipline is abstract; ``name`` must be supplied by the subclass."""
    provider = _ExecCompProvider(
        config, "p", "drag = 2.0 * fltcond_q", VariableSet([DRAG]), VariableSet([DYNAMIC_PRESSURE])
    )
    with pytest.raises(TypeError):
        Discipline(provider, config)


def test_discipline_delegates_provides_and_requires_to_its_provider(config):
    """By default a discipline reports exactly what its provider does."""
    discipline = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    assert discipline.provides().names == {"drag"}
    assert discipline.requires().names == {"fltcond|q"}


def test_discipline_build_delegates_to_the_provider_and_produces_output(config):
    """Building the discipline builds the provider's subsystems and they run."""
    discipline = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])

    prob = om.Problem()
    group = prob.model.add_subsystem("acmodel", om.Group(), promotes=["*"])
    discipline.build(group, num_nodes=4, flight_phase="cruise")
    prob.setup(check=False)
    prob.set_val("fltcond|q", np.full(4, 1000.0), units="N*m**-2")
    prob.run_model()

    assert prob.get_val("drag", units="N") == pytest.approx(np.full(4, 2000.0))


def test_discipline_exposes_provider_and_config(config):
    """Constructor arguments are reachable for reporting."""
    discipline = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    assert discipline.provider.name == "aero"
    assert discipline.config is config


def test_discipline_repr_names_discipline_and_provider(config):
    """The representation identifies both halves of the delegation."""
    discipline = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    assert repr(discipline) == "_NamedDiscipline(name='aero', provider='aero')"


# ==============================================================================
# check_coupling
# ==============================================================================
def test_coupling_accepts_a_set_whose_requirements_are_all_met(config):
    """A discipline set whose internal requirements are satisfied passes quietly."""
    aero = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    prop = _make(config, "propulsion", "thrust = 1.0e5 * throttle", [THRUST], [THROTTLE])

    check_coupling([aero, prop], externally_supplied=[DYNAMIC_PRESSURE, THROTTLE])


def test_coupling_rejects_an_unsatisfied_requirement(config):
    """A required variable with no source is an error, not an input left at its default."""
    aero = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])

    with pytest.raises(CouplingError) as excinfo:
        check_coupling([aero])

    message = str(excinfo.value)
    assert "'fltcond|q' is required by aero" in message
    assert "no discipline provides it" in message


def test_coupling_accepts_a_requirement_the_mission_supplies(config):
    """Declaring a variable as externally supplied is how mission-provided inputs are allowed."""
    aero = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    check_coupling([aero], externally_supplied=[DYNAMIC_PRESSURE])


def test_coupling_rejects_a_variable_provided_twice(config):
    """Two disciplines claiming the same output would make the result order-dependent."""
    first = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    second = _make(config, "aerostructural", "drag = 3.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])

    with pytest.raises(CouplingError) as excinfo:
        check_coupling([first, second], externally_supplied=[DYNAMIC_PRESSURE])

    message = str(excinfo.value)
    assert "'drag' is provided by more than one discipline" in message
    assert "aero" in message and "aerostructural" in message


def test_coupling_reports_every_problem_at_once(config):
    """All problems are listed together; fixing them one traceback at a time invites guesses."""
    first = _make(config, "aero", "drag = 2.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    second = _make(config, "aerostructural", "drag = 3.0 * fltcond_q", [DRAG], [DYNAMIC_PRESSURE])
    third = _make(config, "propulsion", "thrust = 1.0e5 * throttle", [THRUST], [THROTTLE])

    with pytest.raises(CouplingError) as excinfo:
        check_coupling([first, second, third])

    message = str(excinfo.value)
    assert "'drag' is provided by more than one discipline" in message
    assert "'fltcond|q' is required by" in message
    assert "'throttle' is required by propulsion" in message


def test_coupling_allows_one_discipline_to_feed_another(config):
    """A requirement satisfied by a sibling discipline needs no external declaration."""
    prop = _make(config, "propulsion", "thrust = 1.0e5 * throttle", [THRUST], [THROTTLE])
    balance = _make(config, "balance", "drag = 1.0 * thrust", [DRAG], [THRUST])

    check_coupling([prop, balance], externally_supplied=[THROTTLE])


def test_coupling_of_an_empty_set_is_vacuously_satisfied():
    """An empty discipline set has nothing to connect and no problems to report."""
    check_coupling([])
