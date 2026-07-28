"""The sizing loop: closing maximum takeoff weight against the mission that flies it.

An aircraft's takeoff weight determines the fuel it burns, and the fuel it burns determines
its takeoff weight. Sizing means solving that circularity:

.. math::

   \\mathrm{MTOW} = \\mathrm{OEW}(\\mathrm{MTOW}, \\text{geometry}) + W_\\text{payload}
                    + W_\\text{fuel}(\\mathrm{MTOW}, \\text{mission})

:class:`SizingLoop` assembles the model that closes it: the aircraft-scoped disciplines
above, the mission black box below, and a weight balance joining them, all under one Newton
solver.

Which fuel closes the loop matters. It is the fuel burned through the *last* phase fuel
accumulates through -- loiter, at the end of the reserve mission -- not the block fuel at
the end of descent. Using block fuel would size the aircraft to carry no reserves, and the
loop would converge just as readily on the wrong answer.
"""

from __future__ import annotations

from collections.abc import Sequence

import openmdao.api as om
from openconcept.utilities import AddSubtractComp

from cdadt.core.configuration import AircraftConfiguration
from cdadt.core.discipline import Discipline, DisciplineGroup, DisciplineScope
from cdadt.mission.aircraft_model import AircraftModelFactory
from cdadt.mission.blackbox import MissionBlackBox, MissionProfile

__all__ = ["SizingLoop"]


class SizingLoop:
    """Assembles and converges a full mission sizing analysis.

    Parameters
    ----------
    disciplines : sequence of Discipline
        Every discipline in the model, both aircraft-scoped and phase-scoped. They are
        separated by :attr:`~cdadt.core.discipline.Discipline.scope`: aircraft-scoped ones
        are built once above the mission, phase-scoped ones are handed to
        :class:`~cdadt.mission.aircraft_model.AircraftModelFactory` for OpenConcept to
        build inside each phase.
    config : AircraftConfiguration
        Configuration for the aircraft, the mission, and the solver.
    profile : MissionProfile, optional
        The mission to fly. Defaults to
        :meth:`~cdadt.mission.blackbox.MissionProfile.from_config`, so a case is fully
        described by its configuration file.
    num_nodes : int, optional
        Analysis points per mission phase. Must be odd. Default 11.

    Raises
    ------
    MissingConfigurationError
        If a design parameter the disciplines require, or a solver setting, is absent.

    Examples
    --------
    >>> loop = SizingLoop(disciplines, config)
    >>> problem = loop.build()
    >>> loop.converge(problem)
    >>> loop.results(problem)["MTOW"]
    """

    #: Configuration entries the solver reads. No defaults: a tolerance decides when a
    #: result is called converged, so it belongs with the case that reports it.
    SOLVER_SETTINGS = ("solver|maxiter", "solver|atol", "solver|rtol")

    #: Weight variables the loop produces rather than reads from configuration.
    CLOSURE_OUTPUTS = ("ac|weights|MTOW", "ac|weights|W_fuel_max")

    #: Variables the weight closure itself consumes. These are *not* required by any
    #: discipline -- payload appears nowhere in the drag, thrust or empty-weight buildups --
    #: so deriving the design parameters from discipline requirements alone leaves this
    #: input unconnected at OpenMDAO's placeholder default. The loop converges anyway, on an
    #: aircraft carrying one kilogram.
    CLOSURE_INPUTS = ("ac|weights|W_payload",)

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        config: AircraftConfiguration,
        profile: MissionProfile | None = None,
        num_nodes: int = 11,
    ) -> None:
        self._config = config
        self._disciplines = tuple(disciplines)
        self._aircraft = tuple(d for d in self._disciplines if d.scope is DisciplineScope.AIRCRAFT)
        self._phase = tuple(d for d in self._disciplines if d.scope is DisciplineScope.PHASE)
        self._profile = profile if profile is not None else MissionProfile.from_config(config)
        self._num_nodes = num_nodes

        config.require_all(list(self.SOLVER_SETTINGS))
        self._blackbox = MissionBlackBox(
            aircraft_model_class=AircraftModelFactory(self._phase).build(),
            profile=self._profile,
            num_nodes=num_nodes,
        )
        self._design_parameters = self._resolve_design_parameters()

    # -- inspection ---------------------------------------------------------------------

    @property
    def blackbox(self) -> MissionBlackBox:
        """Return the mission black box this loop drives."""
        return self._blackbox

    @property
    def config(self) -> AircraftConfiguration:
        """Return the configuration this loop was built from."""
        return self._config

    @property
    def design_parameters(self) -> tuple[str, ...]:
        """Return the design parameters supplied from configuration, in sorted order.

        These are the ``ac|`` variables the disciplines require that nothing in the model
        produces. They become independent variables at the top of the problem, which is
        what makes them addressable as optimizer design variables.
        """
        return self._design_parameters

    def _resolve_design_parameters(self) -> tuple[str, ...]:
        """Determine which design parameters must come from configuration, and check them.

        Computed rather than listed: a hand-maintained list of design parameters drifts
        from what the disciplines actually read, and the failure is silent because an
        unlisted parameter simply keeps whatever default the component declared.

        Returns
        -------
        tuple of str
            Sorted parameter names.

        Raises
        ------
        MissingConfigurationError
            If any required parameter is absent from the configuration, listing all of
            them.
        """
        required, provided = set(self.CLOSURE_INPUTS), set(self.CLOSURE_OUTPUTS)
        for discipline in self._disciplines:
            provided |= discipline.provides().names
            required |= {name for name in discipline.requires().names if name.startswith("ac|")}

        parameters = tuple(sorted(required - provided))
        self._config.require_all(list(parameters))
        return parameters

    # -- building -----------------------------------------------------------------------

    def build(self, problem: om.Problem | None = None) -> om.Problem:
        """Assemble the sizing problem.

        Parameters
        ----------
        problem : openmdao.api.Problem, optional
            Problem to build into. A new one is created if omitted.

        Returns
        -------
        openmdao.api.Problem
            The problem, set up and ready to converge. Values from the configuration have
            been applied but the mission profile has not; call :meth:`converge`.
        """
        problem = problem if problem is not None else om.Problem()
        model = problem.model

        self._add_design_parameters(model)
        self._add_aircraft_disciplines(model)
        self._add_weight_closure(model)
        self._blackbox.build(model)

        self._configure_solvers(model)
        problem.setup(check=False)
        return problem

    def _add_design_parameters(self, model: om.Group) -> None:
        """Add an independent variable for every configured design parameter.

        Making these explicit outputs, rather than leaving them as OpenMDAO auto-generated
        independent variables, is what lets the optimizer address them by name and what
        puts their configured provenance in one place.
        """
        parameters = model.add_subsystem("design_parameters", om.IndepVarComp(), promotes_outputs=["*"])
        for name in self._design_parameters:
            parameters.add_output(
                name,
                val=self._config.value(name),
                units=self._config.units(name),
            )

    def _add_aircraft_disciplines(self, model: om.Group) -> None:
        """Add the disciplines that describe the airframe rather than a flight condition.

        ``apply_configured_values`` is enabled here because this group sits above the
        mission: it is the highest point at which these inputs are promoted, so seeding
        their configured starting values is correct. Inside a phase it would not be.
        """
        model.add_subsystem(
            "aircraft",
            DisciplineGroup(
                disciplines=self._aircraft,
                config=self._config,
                num_nodes=1,
                flight_phase="aircraft",
                apply_configured_values=True,
            ),
            promotes=["*"],
        )

    def _add_weight_closure(self, model: om.Group) -> None:
        """Add the balance that makes takeoff weight consistent with the fuel burned.

        Total mission fuel is read from the loiter phase, the last phase fuel accumulates
        through, and feeds two places: the takeoff weight sum, and the maximum fuel load
        the empty-weight buildup sizes the fuel system for.
        """
        model.add_subsystem(
            "takeoff_weight",
            AddSubtractComp(
                output_name="ac|weights|MTOW",
                input_names=["ac|weights|OEW", "ac|weights|W_payload", "ac|weights|W_fuel"],
                units="kg",
                lower=1e-6,
            ),
            promotes_inputs=["ac|weights|OEW", "ac|weights|W_payload"],
            promotes_outputs=["ac|weights|MTOW"],
        )

        total_fuel_path = self._blackbox.path("total_fuel")
        model.connect(
            total_fuel_path,
            ["takeoff_weight.ac|weights|W_fuel", "ac|weights|W_fuel_max"],
        )

    def _configure_solvers(self, model: om.Group) -> None:
        """Attach the Newton solver that converges the coupled sizing and mission system.

        ``solve_subsystems`` is required: the mission's own balances -- phase durations,
        throttle, V1 -- must be driven alongside the weight closure rather than after it.
        The bounds-enforcing line search keeps the solver inside the ranges OpenConcept
        declares on its implicit states, several of which are nonphysical if crossed.

        ``err_on_non_converge`` is on and is not configurable. A mission that did not
        converge still produces numbers -- a negative balanced field length, a range that
        misses the one requested -- and those numbers are indistinguishable from results
        unless something refuses to hand them back. This is a correctness stance rather
        than a tuning parameter, so it is not left to a configuration file to get right.
        """
        newton = om.NewtonSolver(solve_subsystems=True)
        newton.options["maxiter"] = int(self._config.scalar("solver|maxiter"))
        newton.options["atol"] = self._config.scalar("solver|atol")
        newton.options["rtol"] = self._config.scalar("solver|rtol")
        newton.options["err_on_non_converge"] = True
        newton.options["iprint"] = -1
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["iprint"] = -1

        model.nonlinear_solver = newton
        model.linear_solver = om.DirectSolver()

    # -- running ------------------------------------------------------------------------

    def seed_initial_state(self, problem: om.Problem) -> None:
        """Set the starting point of the coupled solve from configuration.

        Takeoff weight is an *output* of the weight closure, not an input, so the
        configured value never reaches it through the normal input-default path -- an
        output's starting value comes from whatever its component declared. Left alone the
        Newton solver begins from that placeholder, and the mission it evaluates there is
        far enough from any real aircraft to produce ``inf`` in the atmosphere and engine
        models within a couple of iterations.

        Seeding is not tuning. The configured values are declared as sizing-loop starting
        points with that stated purpose, and the converged answer does not depend on them;
        only whether the solver reaches it does.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem returned by :meth:`build`.
        """
        problem.set_val(
            "ac|weights|MTOW",
            self._config.value("ac|weights|MTOW", units="kg"),
            units="kg",
        )
        problem.set_val(
            "takeoff_weight.ac|weights|W_fuel",
            self._config.value("ac|weights|W_fuel_max", units="kg"),
            units="kg",
        )

    def converge(self, problem: om.Problem, verbose: bool = False) -> None:
        """Seed the starting state, run the continuation schedule, then the design mission.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A problem returned by :meth:`build`.
        verbose : bool, optional
            Print each continuation step. Default ``False``.

        Raises
        ------
        openmdao.core.analysis_error.AnalysisError
            If the Newton solver fails to converge. A non-converged mission still produces
            numbers, and they are indistinguishable from results, so they are not returned.
        """
        self.seed_initial_state(problem)
        self._blackbox.converge(problem, verbose=verbose)

    def results(self, problem: om.Problem) -> dict[str, float]:
        """Return the sized aircraft's top-level results.

        Parameters
        ----------
        problem : openmdao.api.Problem
            A converged problem.

        Returns
        -------
        dict
            Weights in kg, distances in ft, speeds in kn, plus every declared mission
            output. Reading the whole set rather than a chosen few is what makes a run
            report complete by construction.
        """
        results = {
            "MTOW": problem.get_val("ac|weights|MTOW", units="kg").item(),
            "OEW": problem.get_val("ac|weights|OEW", units="kg").item(),
            "MLW": problem.get_val("ac|weights|MLW", units="kg").item(),
            "payload": problem.get_val("ac|weights|W_payload", units="kg").item(),
            "CLmax_cruise": problem.get_val("ac|aero|CLmax_cruise").item(),
            "CLmax_TO": problem.get_val("ac|aero|CLmax_TO").item(),
            "hstab_S_ref": problem.get_val("ac|geom|hstab|S_ref", units="m**2").item(),
            "vstab_S_ref": problem.get_val("ac|geom|vstab|S_ref", units="m**2").item(),
        }
        for name, value in self._blackbox.read_outputs(problem).items():
            results[name] = value.item() if value.size == 1 else value
        return results

    def __repr__(self) -> str:
        """Return a representation naming the discipline split and node count."""
        return (
            f"SizingLoop(aircraft={[d.name for d in self._aircraft]}, "
            f"phase={[d.name for d in self._phase]}, num_nodes={self._num_nodes})"
        )
