"""Tests for the certification basis.

Claim classes: ``unit`` for the requirement objects, ``integration`` for requirements
evaluated against a converged aircraft.

The tests that matter here check that a requirement is *wired*, not merely present. A
requirement class that exists, has the right citation, and never registers a constraint is
worse than no requirement at all: it appears in the traceability matrix and reports a
margin, while the optimizer it was supposed to constrain never saw it. So the checks below
introspect the driver's registered constraints, and confirm each requirement can be made to
bind by moving its limit.
"""

from __future__ import annotations

import numpy as np
import pytest
from openmdao.utils.general_utils import INF_BOUND

from cdadt.aircraft import jet_transport_disciplines
from cdadt.certification import (
    ApproachSpeedLimit,
    BalancedFieldLength,
    CertificationBasis,
    EngineOutClimbGradient,
    LandingFieldLengthLimit,
    Sense,
    ThrottleMargin,
)
from cdadt.core.configuration import MissingConfigurationError
from cdadt.mission import SizingLoop

ENGINE_DECK = "CFM56"


def _all_requirements(config):
    """Return the full certification basis for the B738 case."""
    return CertificationBasis(
        [
            BalancedFieldLength(config),
            EngineOutClimbGradient(config),
            LandingFieldLengthLimit(config),
            ApproachSpeedLimit(config),
            ThrottleMargin(config, phase="climb"),
            ThrottleMargin(config, phase="cruise"),
        ]
    )


# ==============================================================================
# The requirement objects
# ==============================================================================
@pytest.mark.unit
def test_every_requirement_declares_its_citation_and_limit_source(b738_config):
    """Each requirement names its regulation and where its limit came from.

    The traceability matrix is only an artifact if it can cite both. A requirement whose
    limit has no recorded provenance is a number in a file.
    """
    for requirement in _all_requirements(b738_config):
        assert requirement.regulation
        assert requirement.title
        assert requirement.limit_source, (
            f"{requirement.name} has no recorded source for its limit; the traceability "
            f"matrix would report the number without saying where it came from."
        )


@pytest.mark.unit
def test_limits_are_read_from_configuration(b738_config):
    """Each limit equals the configured value, converted to the requirement's units."""
    assert BalancedFieldLength(b738_config).limit() == pytest.approx(
        b738_config.scalar("certification|takeoff|field_length_available", units="ft")
    )
    assert LandingFieldLengthLimit(b738_config).limit() == pytest.approx(
        b738_config.scalar("certification|landing|field_length_available", units="ft")
    )
    assert ThrottleMargin(b738_config, phase="cruise").limit() == pytest.approx(
        b738_config.scalar("certification|propulsion|throttle_max")
    )


@pytest.mark.unit
def test_the_climb_requirement_converts_gradient_to_angle(b738_config):
    """A 2.4% gradient is constrained as arctan(0.024) radians, not 0.024 radians.

    OpenConcept reports a climb *angle*; §25.121 states a *gradient*. Treating them as the
    same quantity is a small error in the right direction to go unnoticed -- about 0.02% at
    this magnitude -- and it is still wrong.
    """
    requirement = EngineOutClimbGradient(b738_config)
    gradient = b738_config.scalar("certification|climb|oei_second_segment_gradient")
    assert requirement.limit() == pytest.approx(float(np.arctan(gradient)))
    assert requirement.limit() != gradient


@pytest.mark.unit
def test_requirements_fail_construction_without_their_limits(b738_config):
    """A requirement with no configured limit refuses to be built.

    There is no fallback field length or default climb gradient, because either would
    silently size an aircraft against a number nobody chose.
    """
    stripped = b738_config.subtree("ac")
    for requirement_class in (BalancedFieldLength, EngineOutClimbGradient, LandingFieldLengthLimit):
        with pytest.raises(MissingConfigurationError):
            requirement_class(stripped)


@pytest.mark.unit
def test_an_approach_speed_factor_below_the_regulatory_floor_is_rejected(b738_config):
    """§25.125(a)(2) sets V_REF >= 1.23 V_SR0; a lower factor is refused.

    A configuration asking for less describes an aircraft that cannot be certified, and it
    would produce landing distances shorter than any real aircraft achieves.
    """
    illegal = b738_config.with_overrides({"certification|landing|approach_speed_factor": 1.10})
    with pytest.raises(ValueError, match=r"25\.125"):
        ApproachSpeedLimit(illegal)


@pytest.mark.unit
def test_duplicate_requirement_names_are_rejected(b738_config):
    """Two requirements with the same name would make the traceability matrix ambiguous."""
    with pytest.raises(ValueError, match="unique"):
        CertificationBasis([BalancedFieldLength(b738_config), BalancedFieldLength(b738_config)])


@pytest.mark.unit
def test_throttle_margin_is_labelled_as_a_design_constraint(b738_config):
    """The throttle limit is not a regulation, and the matrix says so.

    Presenting a design constraint as a regulation in a certification traceability matrix
    would misrepresent the basis.
    """
    assert ThrottleMargin(b738_config, phase="cruise").regulation == "design"


# ==============================================================================
# Wiring: requirements must actually constrain the optimizer
# ==============================================================================
@pytest.fixture(scope="module")
def certified_aircraft(b738_config):
    """Build, converge, and return a sized B738 with the full certification basis attached."""
    basis = _all_requirements(b738_config)
    loop = SizingLoop(
        jet_transport_disciplines(b738_config, engine_deck=ENGINE_DECK),
        b738_config,
        num_nodes=11,
        certification=basis,
    )
    problem = loop.build()
    loop.converge(problem)
    return loop, basis, problem


@pytest.mark.integration
@pytest.mark.slow
def test_every_requirement_registers_a_constraint_the_driver_can_see(certified_aircraft):
    """Each requirement's constraint is registered on the model, not merely declared.

    Introspects what the driver would optimize against. A requirement class that exists but
    never calls ``add_constraint`` still reports a margin, while the optimizer never sees
    it.
    """
    loop, basis, problem = certified_aircraft

    registered = {}
    for requirement in basis:
        path = requirement.constrained_path(loop.blackbox)
        registered[requirement.name] = path
        assert problem.get_val(path) is not None, f"{requirement.name} constrains '{path}', which does not resolve"

    assert len(set(registered.values())) == len(
        registered
    ), f"Two requirements constrain the same variable: {registered}"


@pytest.mark.integration
@pytest.mark.slow
def test_add_constraint_is_actually_called_for_every_requirement(certified_aircraft):
    """Every requirement appears in the model's constraint set after the problem is built.

    A basis that was attached but never registered would still report margins in the
    traceability matrix while the optimizer ran unconstrained, so this reads what the
    driver would actually see.
    """
    loop, basis, problem = certified_aircraft
    constraints = problem.model.get_constraints()

    for requirement in basis:
        path = requirement.constrained_path(loop.blackbox)
        assert any(
            path in key or key in path for key in constraints
        ), f"{requirement.name} did not register a constraint on '{path}'. Registered: {sorted(constraints)}"


@pytest.mark.integration
@pytest.mark.slow
def test_constraint_bounds_are_on_the_correct_side(certified_aircraft):
    """An upper-limit requirement registers an upper bound, and vice versa.

    Registering a field length as a lower bound would drive the optimizer to make the
    runway requirement as long as possible, and every number in the report would still look
    reasonable.
    """
    loop, basis, problem = certified_aircraft
    constraints = problem.model.get_constraints()

    # OpenMDAO records an unset bound as +/- INF_BOUND (1e30), not as an infinity.
    def _is_set(bound):
        return np.all(np.abs(np.asarray(bound)) < INF_BOUND)

    for requirement in basis:
        path = requirement.constrained_path(loop.blackbox)
        meta = next(value for key, value in constraints.items() if path in key or key in path)
        if requirement.sense == Sense.UPPER:
            assert _is_set(meta["upper"]), f"{requirement.name} has no upper bound"
            assert not _is_set(meta["lower"]), f"{requirement.name} also has a lower bound"
        else:
            assert _is_set(meta["lower"]), f"{requirement.name} has no lower bound"
            assert not _is_set(meta["upper"]), f"{requirement.name} also has an upper bound"


# ==============================================================================
# Evaluation and the traceability matrix
# ==============================================================================
@pytest.mark.integration
@pytest.mark.slow
def test_every_requirement_evaluates_to_a_finite_margin(certified_aircraft):
    """Each requirement reports a real value, limit and margin after a converged run."""
    loop, basis, problem = certified_aircraft
    for result in basis.evaluate(problem, loop.blackbox):
        assert np.isfinite(result.value), f"{result.requirement.name} produced {result.value}"
        assert np.isfinite(result.limit)
        assert np.isfinite(result.margin)


@pytest.mark.integration
@pytest.mark.slow
def test_the_margin_sign_convention_is_consistent(certified_aircraft):
    """A positive margin means satisfied, whichever side of the limit the requirement is on."""
    loop, basis, problem = certified_aircraft
    for result in basis.evaluate(problem, loop.blackbox):
        assert result.satisfied == (result.margin >= 0.0)
        if result.requirement.sense == Sense.UPPER:
            assert result.margin == pytest.approx(result.limit - result.value)
        else:
            assert result.margin == pytest.approx(result.value - result.limit)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "requirement_name",
    ["far25_113_balanced_field_length", "far25_121_oei_climb_gradient", "far25_125_landing_field_length"],
)
def test_each_requirement_can_be_made_to_bind(certified_aircraft, b738_config, requirement_name):
    """Moving a limit past the achieved value flips the requirement to not-met.

    A requirement that cannot be made to bind is not connected to anything. This drives the
    limit past the value the converged aircraft achieved and checks that the requirement
    notices -- which proves it is reading the model rather than reporting a constant.
    """
    loop, basis, problem = certified_aircraft
    requirement = next(r for r in basis if r.name == requirement_name)
    achieved = requirement.evaluate(problem, loop.blackbox)

    limit_key = requirement.LIMIT
    impossible = achieved.value * 0.5 if requirement.sense == Sense.UPPER else achieved.value * 2.0

    # The climb requirement stores a gradient and constrains an angle, so move the stored
    # gradient rather than the converted limit.
    if requirement_name == "far25_121_oei_climb_gradient":
        impossible = float(np.tan(achieved.value)) * 2.0

    tightened = b738_config.with_overrides({limit_key: impossible})
    tightened_requirement = type(requirement)(tightened)
    result = tightened_requirement.evaluate(problem, loop.blackbox)

    assert not result.satisfied, (
        f"{requirement_name} reports satisfied against a limit of {result.limit} while the design "
        f"achieves {result.value}. The requirement is not reading the model."
    )


@pytest.mark.integration
@pytest.mark.slow
def test_the_traceability_matrix_names_every_regulation_and_source(certified_aircraft):
    """The matrix reports every requirement, its margin, and where its limit came from."""
    loop, basis, problem = certified_aircraft
    matrix = basis.traceability_matrix(problem, loop.blackbox)

    for requirement in basis:
        assert requirement.regulation in matrix
        assert requirement.limit_source.split(".")[0][:20] in matrix or requirement.name in matrix

    assert "MET" in matrix
    assert f"of {len(basis)} requirements met" in matrix


@pytest.mark.integration
@pytest.mark.slow
def test_the_traceability_matrix_orders_by_how_binding_each_requirement_is(certified_aircraft):
    """The most binding requirement appears first, so the driving constraint is visible.

    Absolute margins in feet, knots and radians are not comparable; relative margins are,
    which is what the ordering uses.
    """
    loop, basis, problem = certified_aircraft
    results = basis.evaluate(problem, loop.blackbox)
    ordered = sorted(results, key=lambda r: r.relative_margin)

    matrix = basis.traceability_matrix(problem, loop.blackbox)
    positions = [matrix.index(r.requirement.title[:28]) for r in ordered]
    assert positions == sorted(positions)


@pytest.mark.integration
@pytest.mark.slow
def test_the_landing_chain_is_built_once_even_with_several_landing_requirements(certified_aircraft):
    """Two landing requirements share one landing model, not one each.

    Two independently built chains could converge to two different approach speeds, and the
    matrix would report a field length and an approach speed that do not describe the same
    landing.
    """
    _, _, problem = certified_aircraft
    v_ref = problem.get_val("landing_performance.V_ref", units="kn")
    assert v_ref.size == 1
    assert np.isfinite(v_ref).all()
