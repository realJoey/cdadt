"""The discipline abstraction.

A discipline is one engineering domain -- aerodynamics, propulsion, weights, geometry,
stability, high lift -- modeled as a class. Each is an :class:`openmdao.api.Group` that owns
the analysis components for its domain and nothing else: it adds its own subsystems, declares
its own promotions, and never reaches into a sibling. Disciplines couple by promoting shared
variable names, which is how OpenMDAO connects them and how OpenConcept's phases connect to
them.

Two scopes, and the distinction is physical rather than organizational:

:class:`AircraftDiscipline`
    Evaluated once for the airframe. Wing MAC, tail areas, operating empty weight and maximum
    lift coefficients are properties of the design; they do not change between cruise and
    descent, and computing them inside each phase would allow them to.
:class:`PhaseDiscipline`
    Evaluated at every analysis node of every mission phase. Drag, thrust and the fuel-burn
    bookkeeping that carries mass through the mission are all functions of the flight
    condition, so OpenConcept instantiates them once per phase, sized to that phase.

There is no global state. Everything a discipline needs arrives as an OpenMDAO option or a
promoted input, which is what lets a single process build a fresh model on every optimizer
iteration.
"""

from __future__ import annotations

from typing import ClassVar

import openmdao.api as om

__all__ = ["AircraftDiscipline", "Discipline", "PhaseDiscipline"]


class Discipline(om.Group):
    """Base class for an engineering discipline.

    Options
    -------
    num_nodes : int
        Number of analysis points this instance is sized for. One for an aircraft-scoped
        discipline; the phase node count for a phase-scoped one.
    flight_phase : str or None
        Name of the mission phase being built, as OpenConcept names it. ``None`` for
        aircraft-scoped disciplines, which are not built inside a phase.
    """

    #: Identifier used as the OpenMDAO subsystem name. Must be a valid OpenMDAO name: no
    #: dots, no spaces.
    discipline_name: ClassVar[str] = "discipline"

    def initialize(self) -> None:
        """Declare the options, matching the signature OpenConcept passes to a phase model."""
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points")
        self.options.declare(
            "flight_phase",
            default=None,
            types=str,
            allow_none=True,
            desc="Name of the mission phase being built",
        )

    def __repr__(self) -> str:
        """Return a representation naming the discipline and its node count."""
        return f"{type(self).__name__}(name={self.discipline_name!r}, num_nodes={self.options['num_nodes']})"


class AircraftDiscipline(Discipline):
    """A discipline whose outputs are properties of the airframe.

    Built once, above the mission, so that every phase sees the same geometry, the same empty
    weight and the same maximum lift coefficients.
    """


class PhaseDiscipline(Discipline):
    """A discipline whose outputs depend on the flight condition.

    Built inside every mission phase by OpenConcept, sized to that phase's node count.
    """

    #: Phases flown with the takeoff flap setting and, therefore, with high-lift drag. The
    #: default is exactly the set OpenConcept's own B738 sizing example uses: the three
    #: ground-roll phases and the rotation. Subclasses widen it -- see
    #: :class:`~cdadt.model.Part25Aerodynamics`, which adds the engine-out climb-angle
    #: condition because 14 CFR 25.121(b) specifies takeoff flaps for it.
    takeoff_configuration_phases: ClassVar[frozenset[str]] = frozenset({"v0v1", "v1v0", "v1vr", "rotate"})

    @property
    def in_takeoff_configuration(self) -> bool:
        """Return whether this instance is being built for a takeoff-configuration phase."""
        return self.options["flight_phase"] in self.takeoff_configuration_phases
