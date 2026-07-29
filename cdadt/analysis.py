"""The sizing analysis: assemble a case, converge it, read it back.

:class:`SizingAnalysis` is the object a case file becomes. It owns the aircraft (which owns the
airframe disciplines), the performance discipline (which owns the initial conditions and the
continuation ladder), and the black box.

The order of operations is the same one OpenConcept's own run scripts follow, and is not
incidental:

1. Build the box, running any registration hooks first, because OpenMDAO requires design
   variables, an objective and constraints to be declared before ``setup``.
2. Write every initial condition -- the design variable values and the mission values alike.
   This is ``set_values(prob, num_nodes)``.
3. Walk the continuation ladder, converging each rung before attempting the next.
4. Converge the design mission.
5. Read every response.

Steps 2 to 4 are what an optimizer repeats -- minus the ladder, which is walked once to
establish the baseline; every later design starts from its predecessor's converged state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import openmdao.api as om

from cdadt.aircraft import Aircraft
from cdadt.blackbox import OpenConceptSizingBox, RunDirectory
from cdadt.certification import CertificationBasis
from cdadt.config import Config
from cdadt.disciplines import Discipline, Performance
from cdadt.results import ResponseCatalog, SizingResults

__all__ = ["SizingAnalysis"]


class SizingAnalysis:
    """A full mission sizing analysis of one aircraft against one mission.

    This is the coordinator: it owns the black box and the disciplines, and it is the only thing
    that talks to both. The disciplines never reach each other.

    Parameters
    ----------
    config : Config
        The case. Everything else is derived from it.
    run : RunDirectory, optional
        Where this study writes its files. ``None`` -- the default -- means the box builds its
        problem with reports off and writes nothing, which is what a test or an interface query
        wants. The command line always supplies one.
    box, aircraft, performance, catalog, certification : optional
        The collaborators, each defaulting to what the ``config`` describes. They are injectable
        so that the coordinator can be driven against a substitute -- a second black box, or a
        stand-in with no model to build -- without going through a case file to do it.

    Examples
    --------
    >>> analysis = SizingAnalysis(Config.from_yaml("cases/b738.yaml"))
    >>> results = analysis.run()
    >>> results["MTOW"]
    78345.6...
    """

    def __init__(
        self,
        config: Config,
        *,
        run: RunDirectory | None = None,
        box: OpenConceptSizingBox | None = None,
        aircraft: Aircraft | None = None,
        performance: Performance | None = None,
        catalog: ResponseCatalog | None = None,
        certification: CertificationBasis | None = None,
    ) -> None:
        self._config = config
        self._box = (
            box
            if box is not None
            else OpenConceptSizingBox(
                model=config.black_box.model,
                num_nodes=config.black_box.num_nodes,
                solver=config.solver.settings(),
                run=run,
            )
        )
        self._aircraft = aircraft if aircraft is not None else Aircraft(config.parameters())
        self._performance = (
            performance if performance is not None else Performance(config.initial_conditions(), config.continuation)
        )
        self._catalog = catalog if catalog is not None else ResponseCatalog()
        self._certification = (
            certification
            if certification is not None
            else CertificationBasis.from_specs(config.constraints, self._catalog)
        )

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
        """The performance discipline, which owns the conditions and the ladder."""
        return self._performance

    @property
    def catalog(self) -> ResponseCatalog:
        """The catalogue mapping response names to black-box paths."""
        return self._catalog

    @property
    def certification(self) -> CertificationBasis:
        """The certification basis this study is judged against.

        Certification is a domain of the study like any other, and this is the class that
        encapsulates it. It is deliberately *not* a
        :class:`~cdadt.disciplines.base.Discipline`: that abstraction means "owns a slice of the
        black box's variable interface", and certification sets nothing and publishes nothing.
        It reads quantities the other disciplines report and judges them, which is a different
        relationship to the box and would be misdescribed by the same base class.

        Owned here rather than by :class:`~cdadt.optimization.Optimizer` so that a sizing run can
        ask whether the aeroplane it converged actually meets its basis, without an optimization
        having to happen first.
        """
        return self._certification

    @property
    def disciplines(self) -> tuple[Discipline, ...]:
        """Every discipline of this study, airframe first and performance last."""
        return (*self._aircraft.disciplines.values(), self._performance)

    # -- running -------------------------------------------------------------------------

    def build(self, register: Sequence[Callable[[om.Group], None]] = ()) -> None:
        """Build the black box and write every initial condition into it.

        Parameters
        ----------
        register : sequence of callable, optional
            Hooks called with the box's group before ``setup``, where design variables, an
            objective and constraints are declared. See :class:`~cdadt.optimization.Optimizer`.

        Raises
        ------
        MissionError
            If a name in the case file is not something the black box publishes, either on its
            own or under the mission path.
        """
        mode = self._config.driver.derivative_mode if self._config.is_optimization else "auto"
        self._box.build(register=register, derivative_mode=mode)

        conditions = self._performance.conditions
        conditions.check(self._box)
        conditions.apply(self._box)

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
        """
        self.build()
        self.converge(verbose=verbose)
        return self.results()

    def __repr__(self) -> str:
        """Return a representation naming the box and the grid."""
        return f"SizingAnalysis({self._box.model_spec!r}, num_nodes={self._box.num_nodes})"
