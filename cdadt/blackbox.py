"""The black box: OpenConcept's full-mission sizing analysis, used as published.

This module is the whole of cdadt's contact with OpenConcept, and it contains no OpenConcept
import. The sizing model is named in the case file as ``module:ClassName`` and loaded by
:func:`importlib.import_module`, so the dependency is data rather than code. A contract test
asserts that no cdadt module imports ``openconcept`` at all.

What that buys is a boundary that cannot erode. cdadt cannot subclass an OpenConcept component
it never imports, cannot re-wire one, and cannot quietly re-implement half of one and call the
result "composition". The only things that cross the boundary are numbers going in and numbers
coming out -- and the name of the model, in a YAML file.

What is inside
--------------

Everything. For the shipped case, ``openconcept.examples.B738_sizing:B738SizingMissionAnalysis``
contains the per-phase aircraft model, the parasite drag buildup, the drag polar, the rubberized
CFM56 deck, the jet-transport empty-weight correlations, the tail volume coefficient sizing, the
maximum lift estimates, the weight closure, and OpenConcept's ``FullMissionWithReserve``
trajectory: balanced-field takeoff, climb, cruise, descent, the Part 25 reserve diversion and
loiter. cdadt computes none of it.

What cdadt supplies
-------------------

The values of the box's independent variables, a mission profile, a Newton solver configuration,
and -- when optimizing -- design variables, an objective and constraints registered on the box's
own group before setup. Registering a design variable on a group is OpenMDAO's public API and
changes no OpenConcept behaviour; it declares which of the box's existing independent variables
a driver may move.

Notes
-----
Attaching the nonlinear and linear solvers is cdadt's job because the box does not attach its
own: OpenConcept's sizing group is written to be dropped into a problem whose run script
supplies them, exactly as its own example does. The settings are configuration, not tuning
constants, and live in the case file.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import openmdao.api as om

__all__ = ["BlackBoxError", "OpenConceptSizingBox", "SolverSettings", "VariableInfo"]


class BlackBoxError(Exception):
    """Raised when the black box cannot be loaded, built, or addressed as asked."""


class SolverSettings:
    """Configuration of the Newton solve that converges the black box.

    Parameters
    ----------
    maxiter : int, optional
        Newton iteration limit. Default 20, as in OpenConcept's own sizing run script.
    atol, rtol : float, optional
        Absolute and relative residual tolerances. Default 1e-9 for both.
    iprint : int, optional
        Solver print level: ``-1`` silent, ``2`` per-iteration. Default ``-1``.
    err_on_non_converge : bool, optional
        Raise instead of returning the state of a failed solve. Default ``True``, and it should
        stay ``True``: a non-converged mission still produces numbers -- a negative field
        length, a range that misses the one requested -- and nothing about them says so.
    """

    __slots__ = ("_atol", "_err_on_non_converge", "_iprint", "_maxiter", "_rtol")

    def __init__(
        self,
        maxiter: int = 20,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        iprint: int = -1,
        err_on_non_converge: bool = True,
    ) -> None:
        self._maxiter = int(maxiter)
        self._atol = float(atol)
        self._rtol = float(rtol)
        self._iprint = int(iprint)
        self._err_on_non_converge = bool(err_on_non_converge)

    @property
    def maxiter(self) -> int:
        """Newton iteration limit."""
        return self._maxiter

    @property
    def atol(self) -> float:
        """Absolute residual tolerance."""
        return self._atol

    @property
    def rtol(self) -> float:
        """Relative residual tolerance."""
        return self._rtol

    @property
    def iprint(self) -> int:
        """Solver print level."""
        return self._iprint

    @property
    def err_on_non_converge(self) -> bool:
        """Whether a failed solve raises."""
        return self._err_on_non_converge

    def __repr__(self) -> str:
        """Return a representation naming the iteration limit and tolerances."""
        return f"SolverSettings(maxiter={self._maxiter}, atol={self._atol}, rtol={self._rtol})"


class VariableInfo:
    """Metadata about one variable of the black box.

    Parameters
    ----------
    name : str
        Promoted name, as cdadt addresses it.
    units : str or None
        Units the box declares.
    shape : tuple of int
        Shape the box declares.
    kind : str
        ``"input"`` for a variable the box takes, ``"output"`` for one it produces.
    """

    __slots__ = ("_kind", "_name", "_shape", "_units")

    def __init__(self, name: str, units: str | None, shape: tuple[int, ...], kind: str) -> None:
        self._name = name
        self._units = units
        self._shape = tuple(shape)
        self._kind = kind

    @property
    def name(self) -> str:
        """Promoted name of the variable."""
        return self._name

    @property
    def units(self) -> str | None:
        """Units the box declares, or ``None`` if dimensionless."""
        return self._units

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape the box declares."""
        return self._shape

    @property
    def kind(self) -> str:
        """``"input"`` or ``"output"``."""
        return self._kind

    @property
    def is_scalar(self) -> bool:
        """Whether the variable holds a single number."""
        return int(np.prod(self._shape)) == 1

    def __repr__(self) -> str:
        """Return a representation naming the variable, units and shape."""
        return f"VariableInfo({self._name!r}, units={self._units!r}, shape={self._shape})"


class OpenConceptSizingBox:
    """An OpenConcept sizing analysis, driven from outside as a black box.

    Parameters
    ----------
    model : str
        The sizing group to load, as ``"module.path:ClassName"``. For the shipped case,
        ``"openconcept.examples.B738_sizing:B738SizingMissionAnalysis"``.
    num_nodes : int
        Analysis points per mission phase. Must be odd: the box integrates fuel burn with
        Simpson's rule, which needs ``2N + 1`` points.
    solver : SolverSettings, optional
        Newton solve configuration. Defaults to OpenConcept's own sizing settings.

    Raises
    ------
    ValueError
        If ``num_nodes`` is even.
    BlackBoxError
        If the model cannot be imported or is not a class.

    Examples
    --------
    >>> box = OpenConceptSizingBox("openconcept.examples.B738_sizing:B738SizingMissionAnalysis", 11)
    >>> box.build()
    >>> box.set("ac|geom|wing|S_ref", 124.6, units="m**2")
    >>> box.run()
    >>> float(box.get("ac|weights|MTOW", units="kg"))
    78345.6...
    """

    #: Tag OpenMDAO puts on the outputs of an ``IndepVarComp``. Those, plus inputs that no
    #: component drives, are exactly the variables a caller may set and a driver may move.
    INDEP_VAR_TAG = "openmdao:indep_var"

    def __init__(self, model: str, num_nodes: int, solver: SolverSettings | None = None) -> None:
        if num_nodes % 2 == 0:
            raise ValueError(
                f"num_nodes must be odd because the box integrates fuel burn with Simpson's rule; got {num_nodes}."
            )
        self._model_spec = str(model)
        self._num_nodes = int(num_nodes)
        self._solver = solver if solver is not None else SolverSettings()
        self._model_class = self._resolve(self._model_spec)
        self._problem: om.Problem | None = None
        self._settable: dict[str, VariableInfo] | None = None
        self._readable: dict[str, VariableInfo] | None = None

    # -- loading -------------------------------------------------------------------------

    @staticmethod
    def _resolve(spec: str) -> type:
        """Return the class named by ``"module.path:ClassName"``.

        Raises
        ------
        BlackBoxError
            If the spec is malformed, the module cannot be imported, the attribute is missing,
            or the attribute is not a class. Each failure names what was tried, because a
            mistyped model in a case file is otherwise an import traceback with no context.
        """
        if spec.count(":") != 1:
            raise BlackBoxError(f"The black-box model must be given as 'module.path:ClassName'; got {spec!r}.")
        module_name, class_name = spec.split(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise BlackBoxError(f"Cannot import the module '{module_name}' named by the black-box model.") from error
        try:
            candidate = getattr(module, class_name)
        except AttributeError as error:
            raise BlackBoxError(f"'{module_name}' has no attribute '{class_name}'.") from error
        if not isinstance(candidate, type):
            raise BlackBoxError(f"'{spec}' names a {type(candidate).__name__}, not a class.")
        return candidate

    @classmethod
    def describe(cls, model: str, num_nodes: int = 3) -> OpenConceptSizingBox:
        """Return a cheaply built box, for reading the interface without running anything.

        Building on the smallest legal grid and with the solver disabled costs a fraction of a
        real setup, and the *names* a box publishes do not depend on the grid -- only their
        shapes do. That is what makes it possible to tell a user their design variable is
        misspelled, with suggestions, before a long optimization starts rather than as an
        OpenMDAO error thrown out of ``setup``.

        Parameters
        ----------
        model : str
            The analysis to load, as ``"module.path:ClassName"``.
        num_nodes : int, optional
            Grid to build on. Default 3, the smallest odd grid.

        Returns
        -------
        OpenConceptSizingBox
            Built, never converged. Its numbers are meaningless; its interface is not.
        """
        box = cls(model, num_nodes, SolverSettings(maxiter=0, iprint=-1, err_on_non_converge=False))
        box.build()
        return box

    @property
    def model_spec(self) -> str:
        """The ``module:Class`` string the box was loaded from."""
        return self._model_spec

    @property
    def model_class(self) -> type:
        """The class the box instantiates. Loaded, never modified and never subclassed."""
        return self._model_class

    @property
    def num_nodes(self) -> int:
        """Analysis points per mission phase."""
        return self._num_nodes

    @property
    def solver(self) -> SolverSettings:
        """The Newton solve configuration."""
        return self._solver

    # -- building ------------------------------------------------------------------------

    def build(
        self,
        register: Sequence[Callable[[om.Group], None]] = (),
        derivative_mode: str = "auto",
        driver: om.Driver | None = None,
    ) -> om.Problem:
        """Instantiate the model, attach solvers, run any registrations, and set up.

        Parameters
        ----------
        register : sequence of callable, optional
            Called with the box's group after the solvers are attached and before ``setup``.
            This is where design variables, an objective and constraints are declared, because
            OpenMDAO requires all three before setup. See
            :class:`~cdadt.optimization.Optimizer`.
        derivative_mode : str, optional
            ``"auto"``, ``"fwd"`` or ``"rev"``. Default ``"auto"``. Forward is usually right
            here: a sizing optimization has a handful of design variables and many vector
            responses.
        driver : openmdao.api.Driver, optional
            Driver to attach before setup. ``None`` leaves OpenMDAO's default.

        Returns
        -------
        openmdao.api.Problem
            The problem, set up but not converged. No mission has been applied yet.
        """
        model = self._model_class(num_nodes=self._num_nodes)
        self._attach_solvers(model)

        problem = om.Problem(model=model, reports=False)
        if driver is not None:
            problem.driver = driver
        for hook in register:
            hook(model)

        problem.setup(check=False, mode=derivative_mode)
        problem.final_setup()

        self._problem = problem
        self._settable = None
        self._readable = None
        return problem

    def _attach_solvers(self, model: om.Group) -> None:
        """Attach the Newton solver and the linear solver the coupled system needs.

        ``solve_subsystems`` is required, not optional. The box's own balances -- phase
        durations against altitude and range targets, throttle against zero acceleration, the
        decision speed against the continue/abort distance difference -- have to be driven
        alongside the weight closure rather than after it. The bounds-enforcing line search
        keeps the solver inside the ranges the box declares on its implicit states, several of
        which are nonphysical if crossed.
        """
        newton = om.NewtonSolver(solve_subsystems=True)
        newton.options["maxiter"] = self._solver.maxiter
        newton.options["atol"] = self._solver.atol
        newton.options["rtol"] = self._solver.rtol
        newton.options["iprint"] = self._solver.iprint
        newton.options["err_on_non_converge"] = self._solver.err_on_non_converge
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["iprint"] = self._solver.iprint

        model.nonlinear_solver = newton
        model.linear_solver = om.DirectSolver()

    @property
    def problem(self) -> om.Problem:
        """The built problem.

        Raises
        ------
        BlackBoxError
            If :meth:`build` has not been called.
        """
        if self._problem is None:
            raise BlackBoxError("The black box has not been built; call build() first.")
        return self._problem

    @property
    def model(self) -> om.Group:
        """The box's own group instance."""
        return self.problem.model

    @property
    def is_built(self) -> bool:
        """Whether :meth:`build` has been called."""
        return self._problem is not None

    # -- the interface -------------------------------------------------------------------

    def settable(self) -> dict[str, VariableInfo]:
        """Return every variable a caller may set, by promoted name.

        These are the box's independent variables: the outputs of its ``IndepVarComp``\\ s and
        the inputs no component drives. They are also exactly the variables that may legally
        become optimizer design variables.

        The set is introspected from the built model, not written down, so a variable added to
        or removed from the box shows up here immediately.

        Returns
        -------
        dict
            ``promoted name -> VariableInfo``.
        """
        if self._settable is None:
            problem = self.problem
            found: dict[str, VariableInfo] = {}
            for _, meta in problem.model.list_outputs(
                tags=self.INDEP_VAR_TAG, val=False, units=True, shape=True, prom_name=True, out_stream=None
            ):
                found[meta["prom_name"]] = VariableInfo(meta["prom_name"], meta["units"], meta["shape"], "output")
            for _, meta in problem.model.list_inputs(
                is_indep_var=True, val=False, units=True, shape=True, prom_name=True, out_stream=None
            ):
                found.setdefault(
                    meta["prom_name"], VariableInfo(meta["prom_name"], meta["units"], meta["shape"], "input")
                )
            self._settable = dict(sorted(found.items()))
        return dict(self._settable)

    def readable(self) -> dict[str, VariableInfo]:
        """Return every output the box produces, by promoted name.

        Returns
        -------
        dict
            ``promoted name -> VariableInfo``. Inputs are omitted: a promoted input name can
            address several components at once and is therefore not unambiguously readable.
            Every quantity a discipline reports is an output.
        """
        if self._readable is None:
            problem = self.problem
            found = {
                meta["prom_name"]: VariableInfo(meta["prom_name"], meta["units"], meta["shape"], "output")
                for _, meta in problem.model.list_outputs(
                    val=False, units=True, shape=True, prom_name=True, out_stream=None
                )
            }
            self._readable = dict(sorted(found.items()))
        return dict(self._readable)

    def has(self, name: str) -> bool:
        """Return whether ``name`` addresses something readable in the box."""
        if not self.is_built:
            return False
        if name in self.readable():
            return True
        try:
            self.problem.get_val(name)
        except (KeyError, RuntimeError, NameError):
            return False
        return True

    def shape_of(self, name: str) -> tuple[int, ...]:
        """Return the shape the box declares for ``name``.

        Used to resample an initial condition written as a pair of endpoints onto the grid the
        box actually declares, which is what ``np.linspace(2300.0, 600.0, num_nodes)`` does by
        hand in OpenConcept's own run script.
        """
        known = self.readable().get(name) or self.settable().get(name)
        if known is not None:
            return known.shape
        return np.atleast_1d(np.asarray(self.get(name))).shape

    def check_settable(self, names: Iterable[str]) -> None:
        """Raise if any of ``names`` is not an independent variable of the box.

        Parameters
        ----------
        names : iterable of str
            Promoted names to check.

        Raises
        ------
        BlackBoxError
            Naming the offenders and, for each, the closest settable variables. Setting a
            computed output would be silently overwritten by the next solve, and declaring one
            as a design variable is an OpenMDAO error thrown far from its cause.
        """
        self._check(names, self.settable(), "independent variables of")

    def check_addressable(self, names: Iterable[str]) -> None:
        """Raise if any of ``names`` is not a variable of the box at all.

        Looser than :meth:`check_settable`, and used for the solver's starting guesses. A
        coupled state such as maximum takeoff weight is a computed *output* -- it is what the
        weight closure solves for -- so it is not an independent variable and can never be a
        design variable. Writing a value onto it before the first solve is nonetheless
        meaningful and often decisive: it is where Newton starts.
        """
        self._check(names, self.readable(), "variables of")

    def _check(self, names: Iterable[str], known: Mapping[str, VariableInfo], what: str) -> None:
        """Raise a suggestion-carrying error for every name outside ``known``."""
        unknown = [name for name in names if name not in known]
        if not unknown:
            return
        details = []
        for name in unknown:
            close = difflib.get_close_matches(name, known, n=3)
            details.append(f"  {name}" + (f"   (did you mean {', '.join(close)}?)" if close else ""))
        raise BlackBoxError(
            f"These are not {what} the black box '{self._model_spec}':\n"
            + "\n".join(details)
            + f"\n{len(known)} are; run 'cdadt inspect <case>' to list them."
        )

    # -- getting and setting -------------------------------------------------------------

    def set(self, name: str, value: Any, units: str | None = None) -> None:
        """Set a variable of the box.

        Parameters
        ----------
        name : str
            Promoted name.
        value : float or array_like
            Value to set, interpreted in ``units``.
        units : str or None, optional
            Units of ``value``. ``None`` means the box's own declared units.

        Raises
        ------
        BlackBoxError
            If the box does not have that variable.
        """
        try:
            self.problem.set_val(name, value, units=units)
        except (KeyError, RuntimeError, NameError, TypeError) as error:
            raise BlackBoxError(f"Cannot set '{name}' on the black box '{self._model_spec}': {error}") from error

    def get(self, name: str, units: str | None = None) -> Any:
        """Read a variable of the box.

        Parameters
        ----------
        name : str
            Promoted name or full path.
        units : str or None, optional
            Units to convert to. ``None`` means the box's own declared units.

        Returns
        -------
        float or numpy.ndarray
            A float for a single-element variable, an array otherwise. Scalars are unwrapped
            because a one-element array propagates into reports and comparisons as
            ``array([1.0])``.

        Raises
        ------
        BlackBoxError
            If the box does not publish that variable.
        """
        try:
            value = self.problem.get_val(name, units=units)
        except (KeyError, RuntimeError, NameError) as error:
            raise BlackBoxError(f"The black box '{self._model_spec}' does not publish '{name}': {error}") from error
        array = np.asarray(value)
        return float(array.reshape(-1)[0]) if array.size == 1 else array

    # -- running -------------------------------------------------------------------------

    def run(self) -> None:
        """Converge the box at the current inputs.

        Raises
        ------
        openmdao.core.analysis_error.AnalysisError
            If the Newton solve fails and ``err_on_non_converge`` is set.
        """
        self.problem.run_model()

    def run_driver(self) -> bool:
        """Run the attached driver.

        Returns
        -------
        bool
            Whether the driver reported success.

        Notes
        -----
        OpenMDAO's ``run_driver`` historically returned a *failure* flag and now returns a
        result object whose truthiness reproduces that, under a deprecation warning. Reading
        ``success`` where it exists keeps cdadt on the supported attribute and, more usefully,
        means the value this method returns says what it means.
        """
        result = self.problem.run_driver()
        success = getattr(result, "success", None)
        return bool(success) if success is not None else not bool(result)

    def __repr__(self) -> str:
        """Return a representation naming the model and the node count."""
        state = "built" if self.is_built else "not built"
        return f"OpenConceptSizingBox({self._model_spec!r}, num_nodes={self._num_nodes}, {state})"
