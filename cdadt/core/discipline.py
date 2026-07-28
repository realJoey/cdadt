"""The discipline abstraction.

A :class:`Discipline` is one engineering domain -- aerodynamics, propulsion, weights,
geometry, stability -- modeled as an object with its own state. It declares what it
provides and what it requires, and delegates its physics to a
:class:`~cdadt.core.provider.Provider`.

Two rules make the abstraction worth having:

**A discipline owns only itself.** :meth:`Discipline.build` adds subsystems to the group it
is given and sets up its own promotions. It never reaches into a sibling discipline, never
inspects the parent, and never mutates anything it did not create. Coupling between
disciplines happens by matching promoted names, and the matching is checked by
:func:`check_coupling` before the model is built.

**No global state.** Everything a discipline needs arrives through its constructor. There
is no registry, no module-level cache, and no import-time side effect. That is what makes
it possible to build several aircraft in one process -- which every sizing loop and every
optimization iteration does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import Enum

import openmdao.api as om

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.provider import Provider
from cdadt.core.variables import Variable, VariableSet

__all__ = [
    "CouplingError",
    "Discipline",
    "DisciplineGroup",
    "DisciplineScope",
    "check_coupling",
    "resolve_input_units",
]


class DisciplineScope(Enum):
    """Where in the model a discipline is built.

    The distinction is physical, not organizational. Some quantities describe the airframe
    and are the same at every point in the mission; others depend on the flight condition
    and must be evaluated at every analysis node of every phase.

    ``AIRCRAFT``
        Built once, above the mission. Geometry, empty weight, tail sizing, and maximum
        lift coefficients are properties of the design, so computing them inside each phase
        would evaluate the same numbers twelve times and, worse, would let them differ
        between phases if an input were wired inconsistently.
    ``PHASE``
        Built inside every mission phase, once per phase, sized to that phase's node count.
        Aerodynamic forces, propulsion, and the fuel-burn bookkeeping that carries mass
        through the mission all depend on the flight condition.
    """

    AIRCRAFT = "aircraft"
    PHASE = "phase"


class CouplingError(RuntimeError):
    """Raised when a set of disciplines cannot be connected.

    Two situations produce it: a discipline requires a variable that nothing in the set
    provides and that the mission does not supply, or two disciplines both claim to
    provide the same variable. Either one would otherwise show up as a silently
    unconnected input sitting at its default value.
    """


class Discipline(ABC):
    """Base class for an engineering discipline.

    Parameters
    ----------
    provider : Provider
        The concrete physics implementation this discipline delegates to.
    config : AircraftConfiguration
        Configuration for the aircraft being modeled. Passed through to the provider and
        available to the discipline for its own bookkeeping.

    Attributes
    ----------
    provider : Provider
        The provider supplied at construction.
    config : AircraftConfiguration
        The configuration supplied at construction.

    Notes
    -----
    Subclasses declare a fixed ``name`` and delegate ``provides``/``requires``/``build`` to
    the provider unless they add coupling of their own. The base class implementations do
    exactly that delegation, so a discipline that is a pure pass-through needs only to
    define :attr:`name`.
    """

    def __init__(self, provider: Provider, config: AircraftConfiguration) -> None:
        self._provider = provider
        self._config = config

    @property
    def provider(self) -> Provider:
        """Return the provider supplying this discipline's physics."""
        return self._provider

    @property
    def config(self) -> AircraftConfiguration:
        """Return the configuration this discipline was constructed with."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the discipline's identifier, e.g. ``"aerodynamics"``.

        Used as the OpenMDAO subsystem name and as the key in reports, so it must be a
        valid OpenMDAO name: no dots, no spaces.
        """

    @property
    def scope(self) -> DisciplineScope:
        """Return where this discipline is built.

        Defaults to :attr:`DisciplineScope.PHASE`, since a discipline that does not depend
        on the flight condition is the exception.
        """
        return DisciplineScope.PHASE

    def provides(self) -> VariableSet:
        """Return the variables this discipline creates as promoted outputs.

        Returns
        -------
        VariableSet
            By default, exactly what the provider provides.
        """
        return self._provider.provides()

    def requires(self) -> VariableSet:
        """Return the variables this discipline consumes as promoted inputs.

        Returns
        -------
        VariableSet
            By default, exactly what the provider requires.
        """
        return self._provider.requires()

    def build(self, group: om.Group, num_nodes: int, flight_phase: str) -> None:
        """Add this discipline's subsystems to ``group``.

        Parameters
        ----------
        group : openmdao.api.Group
            Group to add to. The discipline may add subsystems and set input defaults on
            variables it owns. It must not modify subsystems added by another discipline.
        num_nodes : int
            Number of analysis points in this mission phase.
        flight_phase : str
            Name of the mission phase being built.
        """
        self._provider.build(group, num_nodes=num_nodes, flight_phase=flight_phase)

    def __repr__(self) -> str:
        """Return a representation naming the discipline and its provider."""
        return f"{type(self).__name__}(name={self.name!r}, provider={self._provider.name!r})"


class DisciplineGroup(om.Group):
    """An OpenMDAO group that builds a set of disciplines and reconciles their inputs.

    This is where a discipline set becomes a model. Each discipline is given its own
    subgroup and everything is promoted, so coupling happens by name -- exactly the
    coupling :func:`check_coupling` validated beforehand.

    The group also resolves the input ambiguities that arise from wrapping independently
    written components. See :func:`resolve_input_units` for what is reconciled and why.
    That happens in ``configure()`` rather than ``setup()`` because it is only after the
    children are set up that the group can tell which promoted inputs actually exist --
    and declaring a default for an input that does not exist is itself an error. Whether an
    input exists is genuinely phase-dependent: the drag buildup reads the takeoff flap
    setting in the four takeoff phases and not in the others.

    Parameters
    ----------
    disciplines : sequence of Discipline, optional
        Disciplines to build. If omitted, the class attribute ``_cdadt_disciplines`` is
        used, which is how
        :class:`~cdadt.mission.aircraft_model.AircraftModelFactory` supplies them to a
        class OpenConcept will construct itself.
    config : AircraftConfiguration, optional
        Configuration to read initial input values from. If omitted, the class attribute
        ``_cdadt_config`` is used.

    Notes
    -----
    Three OpenMDAO options are declared. ``num_nodes`` and ``flight_phase`` carry the same
    meaning as OpenConcept's, so a subclass of this group can be handed directly to an
    OpenConcept mission phase.

    ``apply_configured_values`` decides whether configured *values* are seeded alongside
    units, and defaults to ``False``. Seeding a value is only correct where this group is
    the highest point at which the variable is promoted. Inside a mission phase it is not:
    OpenConcept's own components promote design parameters such as
    ``ac|geom|wing|S_ref`` at the phase level with their own placeholder defaults, and a
    value declared inside the aircraft model competes with those rather than overriding
    them. The assembled model sets values above the mission, where the design parameter
    genuinely originates.
    """

    _cdadt_disciplines: tuple[Discipline, ...] = ()
    _cdadt_config: AircraftConfiguration | None = None

    def __init__(
        self,
        disciplines: Sequence[Discipline] | None = None,
        config: AircraftConfiguration | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if disciplines is not None:
            self._cdadt_disciplines = tuple(disciplines)
        if config is not None:
            self._cdadt_config = config

    def initialize(self) -> None:
        """Declare the phase options, matching the ones OpenConcept passes."""
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points in this phase")
        self.options.declare("flight_phase", default=None, types=str, desc="Name of the mission phase")
        self.options.declare(
            "apply_configured_values",
            default=False,
            types=bool,
            desc=(
                "Whether to seed configured initial values as well as units. Only safe when this group is "
                "the highest point at which the variable is promoted; see the class docstring."
            ),
        )

    def setup(self) -> None:
        """Build each discipline into its own promoted subgroup."""
        for discipline in self._cdadt_disciplines:
            subgroup = self.add_subsystem(discipline.name, om.Group(), promotes=["*"])
            discipline.build(
                subgroup,
                num_nodes=self.options["num_nodes"],
                flight_phase=self.options["flight_phase"],
            )

    def configure(self) -> None:
        """Reconcile the units, and optionally the values, of shared inputs."""
        config = self._cdadt_config if self.options["apply_configured_values"] else None
        resolve_input_units(self, self._cdadt_disciplines, config)


def resolve_input_units(
    group: om.Group,
    disciplines: Sequence[Discipline],
    config: AircraftConfiguration | None = None,
) -> None:
    """Declare the units, and where configured the values, of a discipline set's inputs.

    Call this from a group's ``configure()``, not its ``setup()``: it inspects which
    promoted inputs exist, and that is only knowable after the children are set up.

    OpenMDAO refuses to build a model in which two components promote the same input name
    with different units, or with different declared values, and no default is given. That
    is not a hypothetical here: the wrapped OpenConcept components legitimately disagree --
    the empty-weight buildup works in ft and lbm while the geometry components work in m
    and kg -- because each follows the convention of the correlation it implements.

    The information needed to resolve the *units* is already declared:
    :meth:`Discipline.requires` states which units each discipline reads a variable in. So
    the resolution comes from the same statement the contract tests check, rather than from
    a separate list that could drift from it.

    Resolving the *value* is different, because a value is a number and cdadt does not
    invent numbers. Values are taken from ``config`` when the variable is configured, and
    otherwise left alone. If OpenMDAO then reports a value ambiguity, the fix is to
    configure that variable -- which is the correct outcome: the quantity really is an
    input to the model that nothing had specified.

    Only variables the set *consumes without producing* are declared. A variable produced
    inside the set has a source, and its units and value come from that source.

    Parameters
    ----------
    group : openmdao.api.Group
        Group the disciplines were built into.
    disciplines : sequence of Discipline
        The disciplines built into ``group``.
    config : AircraftConfiguration, optional
        Configuration to read initial values from. If omitted, only units are declared.
    """
    provided = set()
    for discipline in disciplines:
        provided |= discipline.provides().names

    declared: dict[str, Variable] = {}
    for discipline in disciplines:
        for variable in discipline.requires():
            if variable.name not in provided:
                declared.setdefault(variable.name, variable)

    present = _promoted_input_names(group)

    for name, variable in declared.items():
        if name not in present:
            continue
        kwargs: dict[str, object] = {}
        if variable.units is not None:
            kwargs["units"] = variable.units
        if config is not None and name in config:
            kwargs["val"] = config.value(name, units=variable.units)
        if kwargs:
            group.set_input_defaults(name, **kwargs)


def _promoted_input_names(group: om.Group) -> set[str]:
    """Return the promoted input names visible at ``group``, callable from ``configure()``.

    A declared requirement may have no corresponding input in a particular build: the drag
    buildup reads the takeoff flap setting in the four takeoff phases and not in the
    others. Declaring an input default for a name with no input is itself an error, so the
    set of names that really exist has to be known before reconciling anything.

    The names are collected from the group's children rather than from the group itself.
    During ``configure()`` the children are set up but the parent's own variable data is
    not yet assembled, so asking the group directly returns nothing. Every child here is
    added with ``promotes=["*"]``, so a child's promoted name is also its name at this
    level.

    Parameters
    ----------
    group : openmdao.api.Group
        Group whose children have completed setup.

    Returns
    -------
    set of str
        Promoted input names.
    """
    names: set[str] = set()
    for subsystem in group._subsystems_myproc:
        metadata = subsystem.get_io_metadata("input", metadata_keys=["units"], return_rel_names=False)
        names |= {meta["prom_name"] for meta in metadata.values()}
    return names


def check_coupling(
    disciplines: Sequence[Discipline],
    externally_supplied: Iterable[Variable] = (),
) -> None:
    """Verify a set of disciplines can be connected before the model is built.

    Two failures are caught here rather than at OpenMDAO setup time, where they would
    appear as an unconnected input quietly holding its declared default:

    * a discipline requires a variable that no other discipline provides and that is not
      supplied externally, and
    * two disciplines both provide the same variable, so which one wins would depend on
      the order they were added.

    Parameters
    ----------
    disciplines : sequence of Discipline
        The disciplines that will be built into one aircraft model.
    externally_supplied : iterable of Variable, optional
        Variables the surrounding model supplies rather than any discipline -- flight
        conditions from the mission phase, throttle, and the like. Declaring them here is
        what distinguishes "supplied by the mission" from "nobody computes this".

    Raises
    ------
    CouplingError
        If any requirement is unsatisfied or any variable is provided more than once. The
        message lists every problem found, not just the first.
    """
    external = VariableSet(externally_supplied)

    provided_by: dict[str, list[str]] = {}
    for discipline in disciplines:
        for variable in discipline.provides():
            provided_by.setdefault(variable.name, []).append(discipline.name)

    duplicates = {name: owners for name, owners in provided_by.items() if len(owners) > 1}

    available = set(provided_by) | set(external.names)
    unsatisfied: dict[str, list[str]] = {}
    for discipline in disciplines:
        for variable in discipline.requires():
            if variable.name not in available:
                unsatisfied.setdefault(variable.name, []).append(discipline.name)

    if not duplicates and not unsatisfied:
        return

    problems = []
    for name, owners in sorted(duplicates.items()):
        problems.append(f"  '{name}' is provided by more than one discipline: {', '.join(sorted(owners))}")
    for name, consumers in sorted(unsatisfied.items()):
        problems.append(
            f"  '{name}' is required by {', '.join(sorted(consumers))} but no discipline provides it "
            f"and it is not declared as externally supplied"
        )
    raise CouplingError(
        "The discipline set cannot be connected:\n" + "\n".join(problems) + "\n"
        "Every promoted input must have exactly one source. Add the missing provider, or "
        "declare the variable in externally_supplied if the mission phase supplies it."
    )
