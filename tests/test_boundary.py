"""Contract: cdadt's dependency boundaries, and the ownership map that spans them.

Everything asserted here is a rule the rest of the project's claims depend on. Together they are
every mandatory rule the project's brief states about a dependency, made enforceable rather than
merely true.

There are three dependencies now -- OpenConcept, openavl, and OpenAeroStruct, which OpenConcept
keeps its wave drag and its own vortex lattice behind -- and each rule below says which of them it
covers. The rules are deliberately not uniform: OpenConcept and openavl may be imported by
``cdadt.adapter`` and nowhere else, while OpenAeroStruct may not be imported by cdadt **at all**,
because cdadt reaches it only through OpenConcept's components.

**cdadt does not import OpenConcept.** Every module is parsed and its imports inspected. cdadt
loads the sizing analysis by name at run time, from a string in the case file, so there is no
import to erode -- and therefore no OpenConcept class it can subclass, no component it can
re-wire, and no way for "compose one part of it" to drift into "re-implement half of it".

**cdadt does not subclass OpenConcept.** Overriding a method changes a component's behaviour
while leaving its source untouched, which no diff of the clone would ever show. Every class
cdadt defines is walked, and its whole inheritance chain checked.

**cdadt does not modify its dependencies.** ``git status`` is run in each installed clone, not only
OpenConcept's -- every number on the vortex-lattice and transonic paths is reproducible only if the
clone that produced it is the published one. Separately, every commit the OpenConcept clone carries
that upstream does not is examined, and the files it touches are compared against the modules cdadt
actually loads.

**cdadt does not copy OpenConcept.** The two rules above are both satisfied by a package that
pasted the code instead: a copy has no import to find, no base class to walk, and leaves the
clone spotless. Distinctive source lines are compared between the two packages.

**cdadt does not patch OpenConcept.** Rebinding an attribute on an imported module changes a
dependency's behaviour while leaving its source, this repository's imports, and every other
check here honest -- and would quietly invalidate the validation test, since the reference run
and the cdadt run would no longer be executing the same code.

**Every input the box accepts has an owner.** The settable set is read off the built model and
every name is routed. A variable cdadt could set that no discipline claims is a hole in the
model of the interface, whether or not any shipped case happens to set it.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import cdadt
from cdadt import Performance
from cdadt.disciplines import AIRCRAFT_DISCIPLINES

PACKAGE = Path(cdadt.__file__).resolve().parent

#: The one package that may name a dependency. Everything cdadt drives is loaded from a string in
#: a case file, so for most of the package this stays impossible -- but installing cdadt's *own*
#: aerodynamics into OpenConcept's mission means composing OpenConcept's blocks around it, and
#: that has to happen somewhere. The brief allows exactly this and no more: *"The wrapper should
#: be the only part of CDADT that directly imports OpenConcept."*
#:
#: Anything added here is a widening of the boundary and should be argued for, not appended to.
ADAPTER_PACKAGE: str = "adapter"


def _is_adapter(module: str) -> bool:
    """Return whether a module is part of the one package allowed to import a dependency."""
    return module == ADAPTER_PACKAGE or module.startswith(f"{ADAPTER_PACKAGE}.")


def _cdadt_sources() -> list[Path]:
    """Return every Python source file in the cdadt package."""
    return sorted(PACKAGE.rglob("*.py"))


def _openconcept_repository() -> Path:
    """Return the root of the OpenConcept clone that is installed."""
    import openconcept

    return Path(openconcept.__file__).resolve().parent.parent


def _git(repository: Path, *arguments: str) -> str:
    """Run git in ``repository`` and return its stdout."""
    return subprocess.run(
        ["git", *arguments], cwd=repository, capture_output=True, text=True, check=True
    ).stdout.strip()


# =============================================================================================
# cdadt does not import OpenConcept
# =============================================================================================


@pytest.mark.contract
def test_no_cdadt_module_outside_the_adapter_imports_openconcept():
    """The dependency is a string in a case file everywhere except the one wrapper package.

    ``cdadt.adapter`` is exempt because it must be: an aircraft model is an OpenMDAO group that
    OpenConcept instantiates, so installing cdadt's aerodynamics means composing OpenConcept's
    propulsion and weight blocks around them. The exemption is one package wide and is what the
    brief asks for. Every other module -- the disciplines, the config, the optimizer, the
    physics in ``cdadt.models`` -- still cannot name the dependency at all.
    """
    offenders = []
    for source in _cdadt_sources():
        module = source.relative_to(PACKAGE).with_suffix("").as_posix().replace("/", ".")
        if _is_adapter(module):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "openconcept" or name.startswith("openconcept."):
                    offenders.append(f"{source.relative_to(PACKAGE.parent)}:{node.lineno} imports {name}")

    assert not offenders, "cdadt must not import OpenConcept outside the adapter:\n" + "\n".join(offenders)


@pytest.mark.contract
def test_no_cdadt_module_anywhere_imports_openaerostruct_directly():
    """OpenAeroStruct is reached *through* OpenConcept, and that distinction is the whole point.

    cdadt uses two OpenAeroStruct-backed things -- OpenConcept's ``WaveDragFromSections`` for the
    transonic drag rise, and its ``VLM`` for validation -- but it imports them from
    ``openconcept.aerodynamics.openaerostruct``, which is an *OpenConcept* module. It never imports
    ``openaerostruct`` itself.

    That keeps the brief's rule intact rather than adding a third exemption to it: all interaction
    with OpenConcept goes through wrapper classes in cdadt, and OpenAeroStruct is something
    OpenConcept depends on, not something cdadt does. This rule has **no adapter exemption** --
    unlike the OpenConcept and openavl rules, it applies to every cdadt module including the
    adapter, because there is no place in cdadt where importing it directly would be right.
    """
    offenders = []
    for source in _cdadt_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "openaerostruct" or name.startswith("openaerostruct."):
                    offenders.append(f"{source.relative_to(PACKAGE.parent)}:{node.lineno} imports {name}")

    assert (
        not offenders
    ), "cdadt must reach OpenAeroStruct through OpenConcept's components, never directly:\n" + "\n".join(offenders)


@pytest.mark.contract
def test_the_physics_cdadt_owns_depends_on_neither_dependency():
    """``cdadt.models`` is the physics, and it must stay writable and testable without either.

    This is the other half of the adapter exemption. Widening the boundary is only defensible if
    it stays where it was widened: the moment a loads model imports OpenConcept or openavl, the
    models stop being independently testable and "cdadt owns its aerodynamics" stops meaning
    anything. Checked separately from the adapter rule so that relaxing one cannot quietly
    relax the other.
    """
    forbidden = ("openconcept", "openavl", "openaerostruct")
    offenders = []
    for source in _cdadt_sources():
        module = source.relative_to(PACKAGE).with_suffix("").as_posix().replace("/", ".")
        if not (module == "models" or module.startswith("models.")):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(name == root or name.startswith(f"{root}.") for root in forbidden):
                    offenders.append(f"{source.relative_to(PACKAGE.parent)}:{node.lineno} imports {name}")

    assert not offenders, "the physics cdadt owns must not import a dependency:\n" + "\n".join(offenders)


@pytest.mark.contract
def test_no_cdadt_class_inherits_from_openconcept():
    """Subclassing would change behaviour invisibly; cdadt drives the box from outside instead."""
    import importlib
    import inspect
    import pkgutil

    modules = [cdadt]
    for info in pkgutil.walk_packages(cdadt.__path__, prefix="cdadt."):
        modules.append(importlib.import_module(info.name))

    offenders = []
    for module in modules:
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not obj.__module__.startswith("cdadt"):
                continue
            for base in inspect.getmro(obj)[1:]:
                if base.__module__.split(".")[0] == "openconcept":
                    offenders.append(f"{module.__name__}.{name} inherits from {base.__module__}.{base.__name__}")

    assert not offenders, "cdadt must not subclass OpenConcept:\n" + "\n".join(offenders)


# =============================================================================================
# cdadt does not modify OpenConcept
# =============================================================================================


@pytest.mark.contract
def test_the_openconcept_working_tree_is_clean():
    """Nothing cdadt does leaves an edit behind in the OpenConcept clone."""
    repository = _openconcept_repository()
    if not (repository / ".git").exists():
        pytest.skip(f"OpenConcept at {repository} is not a git working tree")
    status = _git(repository, "status", "--porcelain")
    assert status == "", f"The OpenConcept clone at {repository} has uncommitted changes:\n{status}"


@pytest.mark.contract
def test_every_dependency_clone_is_clean_not_only_openconcepts():
    """The rule was written for one dependency and cdadt now has three.

    openavl supplies the vortex lattice and OpenAeroStruct the transonic drag rise and the second
    lattice cdadt is validated against. Every number on those paths is only reproducible if the
    clone that produced it is the published one, so "cdadt does not modify its dependencies" has to
    be checked for each of them and not just for the first one it was written for.

    Each is located from its installed package rather than by a hardcoded path, and skipped rather
    than failed when it is absent, since both are optional extras.
    """
    dirty = []
    for package_name in ("openavl", "openaerostruct"):
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            continue
        for directory in Path(package.__file__).resolve().parents:
            if (directory / ".git").exists():
                status = _git(directory, "status", "--porcelain")
                if status:
                    dirty.append(f"{package_name} at {directory}:\n{status}")
                break

    assert not dirty, "a dependency clone cdadt reads has uncommitted changes:\n" + "\n".join(dirty)


@pytest.mark.contract
def test_no_locally_committed_openconcept_change_touches_a_module_cdadt_loads(built_box):
    """A clean tree says nothing about local *commits*; this checks those.

    The installed clone may carry commits that upstream does not -- compatibility fixes made in
    other work. That is only acceptable if none of them touches a module the black box actually
    loads. Which modules those are is taken from ``sys.modules`` after a box has been built, so
    it is measured rather than assumed.
    """
    repository = _openconcept_repository()
    if not (repository / ".git").exists():
        pytest.skip(f"OpenConcept at {repository} is not a git working tree")
    try:
        local_commits = _git(repository, "log", "--format=%H", "origin/main..HEAD")
    except subprocess.CalledProcessError:
        pytest.skip("No origin/main to compare the OpenConcept clone against")
    if not local_commits:
        return

    touched = set(_git(repository, "diff", "--name-only", "origin/main...HEAD").splitlines())
    loaded = {
        Path(module.__file__).resolve().relative_to(repository).as_posix()
        for name, module in sys.modules.items()
        if name.split(".")[0] == "openconcept" and getattr(module, "__file__", None)
    }
    overlap = sorted(touched & loaded)
    assert not overlap, (
        "The OpenConcept clone carries local commits touching modules cdadt loads, so its "
        "results are not those of the published library:\n  " + "\n  ".join(overlap)
    )


# =============================================================================================
# cdadt does not copy OpenConcept, and does not reach in and change it
# =============================================================================================


#: Shortest line worth comparing. Below this, a shared line is punctuation and API boilerplate --
#: ``newton.linesearch = om.BoundsEnforceLS()`` is cdadt configuring OpenMDAO the way any caller
#: would, not a lifted implementation.
DISTINCTIVE_LINE = 25

#: Longest run of *consecutive* cdadt lines that may also appear in OpenConcept.
#:
#: This measures runs rather than a total count, and the reason is worth stating because the
#: first version of this guard counted totals and measured the wrong thing. Composing
#: OpenConcept's components means writing OpenConcept's variable names --
#: ``promotes_inputs=["throttle", "fltcond|h", ...]`` is not a copy, it is the interface, and
#: there is no other way to spell it. Those matches are isolated and scattered.
#:
#: Copied source looks completely different: it arrives in unbroken stretches. Both ends are
#: measured rather than guessed. Honest composition peaks at **4** -- and that is
#: ``cdadt/mission.py`` deliberately quoting four consecutive lines of ``B738.py`` in its
#: docstring to say which reference function it answers to. ``cdadt/adapter/aircraft.py``, which
#: composes OpenConcept's components directly, reaches only **2**.
#:
#: Copying one real module gives 10 to 26, over six sampled across the library and the examples.
#: Eight sits between the two with margin either way.
MAX_SHARED_RUN = 8


def _longest_shared_run(source: Path, theirs: set[str]) -> tuple[int, str]:
    """Return the longest run of consecutive distinctive lines in ``source`` also in ``theirs``.

    Blank lines and comments neither extend nor break a run: a copy with the comments stripped is
    still a copy, and reformatting should not defeat the measurement.
    """
    longest, running, where = 0, 0, ""
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _is_distinctive(line) and line in theirs:
            running += 1
            if running > longest:
                longest, where = running, line
        else:
            running = 0
    return longest, where


def _is_distinctive(line: str) -> bool:
    """Return whether a line is substantial enough that sharing it means anything."""
    return len(line) >= DISTINCTIVE_LINE and not line.startswith(("#", '"', "'", "from ", "import ", "@"))


def _distinctive_lines(root: Path) -> set[str]:
    """Return the source lines of ``root`` that are long enough to be worth comparing.

    Comments, docstring delimiters, imports and decorators are dropped: they are shared between
    any two Python projects and would drown the signal a real copy would produce.
    """
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if len(line) >= DISTINCTIVE_LINE and not line.startswith(("#", '"', "'", "from ", "import ", "@")):
                found.add(line)
    return found


@pytest.mark.contract
def test_cdadt_does_not_duplicate_openconcept_source():
    """No component, correlation or model is copied across the boundary.

    "Do not import it" and "do not subclass it" are both satisfied by a package that pasted the
    code instead, and neither of the other tests here would notice: a copy has no import to find
    and no base class to walk. Nor would a diff of the clone, since the clone is untouched.

    What is measured is the longest **consecutive** run, not a total. Isolated shared lines are
    unavoidable and meaningless: :mod:`cdadt.adapter` composes OpenConcept's components, and
    doing that means naming OpenConcept's variables exactly as OpenConcept spells them. A copy is
    a different shape entirely -- an unbroken stretch of matching lines.
    """
    theirs = _distinctive_lines(_openconcept_repository() / "openconcept")

    offenders = []
    for source in _cdadt_sources():
        run, line = _longest_shared_run(source, theirs)
        if run > MAX_SHARED_RUN:
            offenders.append(f"  {source.relative_to(PACKAGE.parent)}: {run} consecutive lines, ending {line[:60]!r}")

    assert not offenders, (
        "cdadt appears to contain copied OpenConcept source -- a run of consecutive identical "
        "lines is what a paste looks like:\n" + "\n".join(offenders)
    )


@pytest.mark.contract
def test_no_cdadt_module_rebinds_an_attribute_on_an_imported_module():
    """cdadt never monkey-patches: not OpenConcept, and not anything else it depends on.

    Patching changes a dependency's behaviour at run time while leaving both its source and this
    repository's imports honest, so it is invisible to every other check here -- and it would
    make the validation test meaningless, because the reference run and the cdadt run would no
    longer be executing the same code.

    Every assignment in the package is parsed. What is refused is a write *through* a name the
    module did not define: ``some_module.attribute = ...`` and ``setattr(some_module, ...)``.
    Writes to ``self`` and ``cls`` are how objects hold their own state and are not that.
    """
    offenders = []
    for path in _cdadt_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        }
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else []
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in imported
                ):
                    offenders.append(f"  {path.name}:{node.lineno}: {ast.unparse(target)} = ...")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in imported
            ):
                offenders.append(f"  {path.name}:{node.lineno}: setattr({ast.unparse(node.args[0])}, ...)")

    assert not offenders, "cdadt must not patch a module it imports:\n" + "\n".join(offenders)


# =============================================================================================
# The ownership map is total and disjoint
# =============================================================================================


@pytest.mark.contract
def test_every_settable_variable_of_the_box_has_exactly_one_owner(built_box):
    """No input cdadt can set is unowned, and none is owned twice."""
    # Ownership is a class-level property, so the classes answer it without being instantiated.
    disciplines = (*AIRCRAFT_DISCIPLINES, Performance)

    unowned, contested = [], []
    for name in built_box.settable():
        owners = [d.discipline_name for d in disciplines if d.owns(name)]
        if not owners:
            unowned.append(name)
        elif len(owners) > 1:
            contested.append(f"{name}: {owners}")

    assert not unowned, "These settable variables of the box are owned by no discipline:\n  " + "\n  ".join(unowned)
    assert not contested, "These settable variables are claimed by more than one discipline:\n  " + "\n  ".join(
        contested
    )


@pytest.mark.contract
def test_every_reported_response_exists_in_the_box(built_box):
    """A discipline that reports a path the box does not publish would report nothing quietly."""
    missing = []
    for discipline in (*AIRCRAFT_DISCIPLINES, Performance):
        for response in discipline.reported:
            if response.optional:
                continue
            if not built_box.has(response.path):
                missing.append(f"{discipline.discipline_name}.{response.name} -> {response.path}")

    assert not missing, "These required responses are not published by the box:\n  " + "\n  ".join(missing)


@pytest.mark.contract
def test_the_optional_responses_of_the_shipped_case_are_all_available(built_box, converged_analysis):
    """Which optional responses the shipped box publishes, stated exactly rather than assumed.

    This used to assert that *nothing* was unavailable, which was true when every optional response
    happened to exist in OpenConcept's own sizing group. It stopped being true the moment cdadt
    published a response that group does not compose, and the honest fix is to name what is missing
    rather than to drop the check.

    ``wing_span`` is the case. cdadt's own analysis group composes OpenConcept's ``WingSpan``, so a
    study on that group can constrain the span against a gate limit. ``B738SizingMissionAnalysis``
    never composes it -- nothing in it needs a span -- so on the shipped case the response is
    genuinely absent, and reporting it as absent is the mechanism working rather than failing.
    """
    unavailable = converged_analysis.aircraft.missing(built_box)
    assert unavailable == {"geometry": ("wing_span",)}, (
        "the shipped black box's optional responses have changed; if that is intended, say so here "
        f"rather than leaving the list to drift: {unavailable}"
    )


# =============================================================================================
# There is no shared mutable state
# =============================================================================================


@pytest.mark.contract
def test_cdadt_has_no_module_level_mutable_state():
    """No module in the package binds a mutable object at import time.

    ``__all__`` is exempt: it is a list by language convention, is never mutated, and exists to
    describe the module rather than to hold state.
    """
    offenders = []
    for source in _cdadt_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else ([node.target] if isinstance(node, ast.AnnAssign) else [])
            )
            for target in targets:
                if not isinstance(target, ast.Name) or target.id == "__all__":
                    continue
                value = node.value
                mutable_literal = isinstance(
                    value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp | ast.SetComp
                )
                mutable_call = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in {"dict", "list", "set"}
                )
                if mutable_literal or mutable_call:
                    offenders.append(f"{source.relative_to(PACKAGE.parent)}:{node.lineno} {target.id}")

    assert not offenders, "cdadt must hold no module-level mutable state:\n  " + "\n  ".join(offenders)


@pytest.mark.contract
def test_no_cdadt_function_carries_a_cache_that_outlives_its_caller():
    """A memoising decorator is module-level mutable state, and the other guards cannot see it.

    This was a real hole rather than a hypothetical one. ``cdadt.adapter.lattice`` memoised its
    lattice solves with ``@lru_cache`` on a module-level function, which is the obvious way to cache
    a geometry-keyed solve -- and it survived every check on this page, because both of the guards
    above look for *assignments* of mutable literals and a decorator assigns nothing. Meanwhile
    :doc:`/architecture` claimed the suite enforced that cdadt has no shared mutable state.

    What made it more than a technicality: the cache persisted across every ``Problem`` in the
    process, anyone who could import the module could empty it through ``cache_clear()``, and two
    independently constructed models were handed the same object. The replacement is
    :class:`~cdadt.adapter.lattice.LatticeLibrary` -- the same memo, owned by an object whose
    lifetime the caller chooses and injects.
    """
    caching = {"lru_cache", "cache", "cached_property"}
    offenders = []
    for source in _cdadt_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                # Both `@lru_cache` and `@lru_cache(maxsize=...)`, and dotted forms of either.
                reference = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = reference.attr if isinstance(reference, ast.Attribute) else getattr(reference, "id", "")
                if name in caching:
                    offenders.append(
                        f"{source.relative_to(PACKAGE.parent)}:{node.lineno} {node.name} is decorated with {name}"
                    )

    assert not offenders, (
        "a cache on a function outlives every caller and is reachable by anyone who can import the "
        "module; give it to an object whose lifetime somebody owns instead:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.contract
def test_cdadt_has_no_class_level_mutable_state():
    """No class in the package carries a mutable class attribute.

    This is the check that removed the last one. An early design gave a ``Requirement`` class a
    class-level ``registry`` dict, populated by ``__init_subclass__``. It was convenient and it
    was shared mutable state: two studies in one process shared it, defining a class anywhere
    mutated it, and what a case file resolved to depended on what had been imported. That whole
    design is gone -- constraints are built from the case file into a
    :class:`~cdadt.certification.CertificationBasis`, which holds them as instance state -- and
    ``test_constraints_are_built_from_the_case_file_rather_than_a_registry`` below asserts the
    old names stay gone.

    Immutable class attributes -- the ownership patterns, the response tuples, the panel
    definitions -- are the intended way to declare what a class *is*, and are unaffected.
    """
    import importlib
    import inspect
    import pkgutil

    modules = [cdadt]
    for info in pkgutil.walk_packages(cdadt.__path__, prefix="cdadt."):
        modules.append(importlib.import_module(info.name))

    offenders, seen = [], set()
    for module in modules:
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not obj.__module__.startswith("cdadt") or obj in seen:
                continue
            seen.add(obj)
            for attribute, value in vars(obj).items():
                if attribute.startswith("__"):
                    continue
                if isinstance(value, dict | list | set):
                    offenders.append(f"{obj.__module__}.{name}.{attribute} = {type(value).__name__}")

    assert not offenders, "cdadt must hold no class-level mutable state:\n  " + "\n  ".join(offenders)


@pytest.mark.contract
def test_constraints_are_built_from_the_case_file_rather_than_a_registry():
    """There is no class-level registry of constraint types, and there is nothing to mutate.

    An earlier design resolved a case file's ``type:`` through a dictionary populated at import
    time by ``__init_subclass__``. It was shared mutable state: two studies in one process shared
    it, and what a case file resolved to depended on what had been imported. Constraints are now
    plain named bounds built straight from the case file, so the question does not arise.
    """
    from cdadt import CertificationBasis, ConstraintSpec
    from cdadt.config import Bounds

    basis = CertificationBasis.from_specs([ConstraintSpec("takeoff_field_length", Bounds(upper=8000.0), units="ft")])
    assert len(basis) == 1
    assert basis.constraints[0].path == "mission.bfl.distance_continue"

    import cdadt.certification as certification

    assert not hasattr(certification, "Requirement"), "the requirement registry is back"
    assert not hasattr(certification, "SHIPPED_REQUIREMENTS")
