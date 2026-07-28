"""The sizing analysis: assemble a case, converge it, read it back.

:class:`SizingAnalysis` is the object a case file becomes. It owns the aircraft (which owns the
airframe disciplines), the performance discipline (which owns the mission), and the black box.
It builds the box, writes every discipline's state into it, walks the continuation ladder to
convergence, and reads every discipline's responses back.

The order of operations is not incidental:

1. Build the box, running any registration hooks first, because OpenMDAO requires design
   variables, an objective and constraints to be declared before ``setup``.
2. Write the aircraft parameters and the initial guesses, once. They do not change during the
   continuation.
3. Walk the continuation ladder. Each rung is converged before the next is attempted, so the
   solver always starts from a converged neighbour.
4. Converge the design mission.
5. Read every response.

Steps 2 to 4 are what an optimizer repeats -- minus the ladder, which is walked once to
establish the baseline; every later design starts from its predecessor's converged state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import openmdao.api as om

from cdadt.aircraft import Aircraft
from cdadt.blackbox import OpenConceptSizingBox
from cdadt.config import Config
from cdadt.disciplines import Discipline, Performance
from cdadt.results import ResponseCatalog, SizingResults

__all__ = ["SizingAnalysis"]


class SizingAnalysis:
    """A full mission sizing analysis of one aircraft against one mission.

    Parameters
    ----------
    config : Config
        The case. Everything else is derived from it.

    Examples
    --------
    >>> analysis = SizingAnalysis(Config.from_yaml("cases/b738.yaml"))
    >>> results = analysis.run()
    >>> results["MTOW"]
    78345.6...
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._box = OpenConceptSizingBox(
            model=config.black_box.model,
            num_nodes=config.black_box.num_nodes,
            solver=config.solver.settings(),
        )
        self._aircraft = Aircraft(config.aircraft)
        self._performance = Performance(config.mission.profile(), config.black_box.num_nodes)
        self._catalog = ResponseCatalog()

    # -- state ---------------------------------------------------------------------------

    @property
    def config(self) -> Config:
        """The case this analysis was built from."""
        return self._config

    @property
    def box(self) -> OpenConceptSizingBox:
        """The black box."""
        return self._box

    @property
    def aircraft(self) -> Aircraft:
        """The airframe disciplines."""
        return self._aircraft

    @property
    def performance(self) -> Performance:
        """The performance discipline, which owns the mission."""
        return self._performance

    @property
    def catalog(self) -> ResponseCatalog:
        """The catalogue mapping response names to black-box paths."""
        return self._catalog

    @property
    def disciplines(self) -> tuple[Discipline, ...]:
        """Every discipline of this study, airframe first and performance last."""
        return (*self._aircraft.disciplines.values(), self._performance)

    # -- running -------------------------------------------------------------------------

    def build(self, register: Sequence[Callable[[om.Group], None]] = ()) -> None:
        """Build the black box and write the aircraft into it.

        Parameters
        ----------
        register : sequence of callable, optional
            Hooks called with the box's group before ``setup``, where design variables, an
            objective and constraints are declared. See :class:`~cdadt.optimization.Optimizer`.

        Raises
        ------
        BlackBoxError
            If a parameter in the case file is not a variable this black box lets a caller set.
            The check is made against the built model rather than a list, and names every
            offender at once with suggestions.
        """
        mode = self._config.optimization.derivative_mode if self._config.is_optimization else "auto"
        self._box.build(register=register, derivative_mode=mode)

        self._box.check_settable([p.name for p in self._config.aircraft] + list(self._config.mission.parameter_names))
        # Starting guesses are held to a looser standard on purpose: the states that matter
        # most -- maximum takeoff weight, the fuel load -- are outputs the box solves for, not
        # inputs it takes, and seeding them is writing where Newton starts.
        self._box.check_addressable([p.name for p in self._config.initial_guesses])

        self._aircraft.apply(self._box)
        for guess in self._config.initial_guesses:
            self._box.set(guess.name, guess.value, units=guess.units)

    def converge(self, verbose: bool = False) -> None:
        """Walk the continuation ladder and converge the design mission.

        Raises
        ------
        openmdao.core.analysis_error.AnalysisError
            If a Newton solve fails and the case asked for that to be an error.
        """
        self._performance.converge(self._box, verbose=verbose)

    def results(self) -> SizingResults:
        """Read every discipline's responses out of the converged box."""
        return SizingResults.collect(self._box, self.disciplines)

    def run(self, verbose: bool = False) -> SizingResults:
        """Build, converge and read.

        Parameters
        ----------
        verbose : bool, optional
            Print each continuation step as it runs. Default ``False``.

        Returns
        -------
        SizingResults
            Every response of every discipline.
        """
        self.build()
        self.converge(verbose=verbose)
        return self.results()

    def __repr__(self) -> str:
        """Return a representation naming the box and the grid."""
        return f"SizingAnalysis({self._box.model_spec!r}, num_nodes={self._box.num_nodes})"
