"""Building the aircraft model class that OpenConcept instantiates.

OpenConcept's mission phases do not accept an aircraft model *object*. They accept a class,
and construct it themselves with a fixed signature:

.. code-block:: python

   self.options["aircraft_model"](num_nodes=nn, flight_phase=self.options["flight_phase"])

There is no argument through which a configured set of :class:`~cdadt.core.discipline.Discipline`
instances could be passed. The obvious workarounds are both bad:

* a module-level registry the class reads at ``setup()`` time is global mutable state, and
  every sizing iteration and optimizer function evaluation builds aircraft models, so the
  results would depend on evaluation order; and
* ``functools.partial`` works -- OpenConcept only calls the option -- but a partial is not
  a class, so the model tree becomes harder to inspect and the failure mode if OpenConcept
  ever adds a ``types=`` check is a confusing one.

:class:`AircraftModelFactory` instead builds a new class per discipline set, carrying the
disciplines as a class attribute. The result is a genuine ``om.Group`` subclass with exactly
the signature OpenConcept expects, holding no state shared with any other model.
"""

from __future__ import annotations

from collections.abc import Sequence

import openmdao.api as om

from cdadt.core.discipline import CouplingError, Discipline, check_coupling
from cdadt.mission.contract import (
    AIRCRAFT_MODEL_OUTPUTS,
    PHASE_SPECS_BY_NAME,
    phase_supplied_inputs,
)

__all__ = ["AircraftModelFactory", "CdadtAircraftModel"]


class CdadtAircraftModel(om.Group):
    """Base class for the group OpenConcept instantiates inside every mission phase.

    This class is not used directly. :meth:`AircraftModelFactory.build` produces a subclass
    with ``_cdadt_disciplines`` bound, and OpenConcept instantiates that.

    Notes
    -----
    Two OpenMDAO options are declared, both set by OpenConcept at construction:
    ``num_nodes``, the number of analysis points in the phase, and ``flight_phase``, the
    name of the phase being built.

    The subsystems are added by the disciplines themselves. This class contributes no
    physics of its own; it exists to satisfy OpenConcept's construction signature and to
    give the disciplines a group to build into.
    """

    _cdadt_disciplines: tuple[Discipline, ...] = ()

    def initialize(self) -> None:
        """Declare the options OpenConcept passes at construction."""
        self.options.declare("num_nodes", default=1, types=int, desc="Number of analysis points in this phase")
        self.options.declare("flight_phase", default=None, types=str, desc="Name of the mission phase")

    def setup(self) -> None:
        """Build each discipline into this group, promoting everything.

        Promotion is total in both directions, matching how OpenConcept promotes the
        aircraft model itself (``promotes_inputs=["*"], promotes_outputs=["*"]``). Coupling
        between disciplines and to the surrounding phase is therefore by name, which is
        exactly what :func:`~cdadt.core.discipline.check_coupling` validated before the
        model was built.
        """
        num_nodes = self.options["num_nodes"]
        flight_phase = self.options["flight_phase"]

        for discipline in self._cdadt_disciplines:
            subgroup = self.add_subsystem(discipline.name, om.Group(), promotes=["*"])
            discipline.build(subgroup, num_nodes=num_nodes, flight_phase=flight_phase)


class AircraftModelFactory:
    """Builds the aircraft-model class for a configured set of disciplines.

    Parameters
    ----------
    disciplines : sequence of Discipline
        The disciplines that together turn flight conditions into thrust, drag, and mass.

    Raises
    ------
    CouplingError
        If the discipline set cannot be connected -- a requirement with no source, or a
        variable provided twice. Checked against the variables OpenConcept supplies in each
        phase, so a discipline that consumes a flight condition unavailable in a particular
        phase is rejected here rather than left as an unconnected input at solve time.
    ValueError
        If the disciplines do not, between them, produce every variable in
        :data:`~cdadt.mission.contract.AIRCRAFT_MODEL_OUTPUTS`.

    Examples
    --------
    >>> factory = AircraftModelFactory([geometry, aerodynamics, propulsion, weights])
    >>> model_class = factory.build()
    >>> mission = FullMissionWithReserve(num_nodes=11, aircraft_model=model_class)
    """

    def __init__(self, disciplines: Sequence[Discipline]) -> None:
        self._disciplines = tuple(disciplines)
        self._validate()

    @property
    def disciplines(self) -> tuple[Discipline, ...]:
        """Return the disciplines this factory will build into the aircraft model."""
        return self._disciplines

    def build(self) -> type[CdadtAircraftModel]:
        """Return a class OpenConcept can instantiate as ``aircraft_model``.

        Returns
        -------
        type
            A new subclass of :class:`CdadtAircraftModel` with this factory's disciplines
            bound as a class attribute. Each call returns a distinct class, so two
            factories never share state.
        """
        discipline_names = "_".join(d.name for d in self._disciplines) or "empty"
        return type(
            f"CdadtAircraftModel_{discipline_names}",
            (CdadtAircraftModel,),
            {
                "_cdadt_disciplines": self._disciplines,
                "__doc__": (
                    "cdadt aircraft model built from disciplines: "
                    + ", ".join(d.name for d in self._disciplines)
                    + ". Generated by AircraftModelFactory; see cdadt.mission.aircraft_model."
                ),
            },
        )

    def _validate(self) -> None:
        """Check the discipline set against the contract, for every phase.

        Two things are checked, and both are checked per phase because the variables
        OpenConcept supplies differ between a ground roll and a steady cruise:

        * every required variable has a source, and nothing is provided twice
          (:func:`~cdadt.core.discipline.check_coupling`), and
        * the set produces thrust, drag, and mass.
        """
        provided = set()
        for discipline in self._disciplines:
            provided |= discipline.provides().names

        missing_outputs = sorted(AIRCRAFT_MODEL_OUTPUTS.names - provided)
        if missing_outputs:
            raise ValueError(
                f"The discipline set does not produce {missing_outputs}, which OpenConcept requires from "
                f"every aircraft model. Produced: {sorted(provided)}. Without these, the mission's "
                f"acceleration and steady-flight balances read unconnected inputs holding their defaults."
            )

        # Design parameters under 'ac|' are promoted from the top level of the mission, not
        # produced by a discipline or a phase, so they are available everywhere.
        design_parameters = [
            variable
            for discipline in self._disciplines
            for variable in discipline.requires()
            if variable.name.startswith("ac|")
        ]

        for phase_name in PHASE_SPECS_BY_NAME:
            supplied = phase_supplied_inputs(phase_name)
            try:
                check_coupling(self._disciplines, externally_supplied=[*supplied, *design_parameters])
            except CouplingError as err:
                raise CouplingError(
                    f"The discipline set cannot be built into the '{phase_name}' phase.\n{err}\n"
                    f"Variables OpenConcept supplies in this phase: {sorted(supplied.names)}"
                ) from err

    def __repr__(self) -> str:
        """Return a representation listing the discipline names."""
        return f"AircraftModelFactory({[d.name for d in self._disciplines]})"
