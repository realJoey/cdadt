"""Contract: the cdadt/OpenConcept boundary, and the ownership map that spans it.

Six things are asserted here, each of which the rest of the project's claims depend on. Together
they are every mandatory rule the project's brief states about the dependency, made enforceable
rather than merely true.

**cdadt does not import OpenConcept.** Every module is parsed and its imports inspected. cdadt
loads the sizing analysis by name at run time, from a string in the case file, so there is no
import to erode -- and therefore no OpenConcept class it can subclass, no component it can
re-wire, and no way for "compose one part of it" to drift into "re-implement half of it".

**cdadt does not subclass OpenConcept.** Overriding a method changes a component's behaviour
while leaving its source untouched, which no diff of the clone would ever show. Every class
cdadt defines is walked, and its whole inheritance chain checked.

**cdadt does not modify OpenConcept.** ``git status`` is run in the OpenConcept working tree.
Separately, every commit the clone carries that upstream does not is examined, and the files it
touches are compared against the modules cdadt actually loads.

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
import subprocess
import sys
from pathlib import Path

import pytest

import cdadt
from cdadt import Performance
from cdadt.disciplines import AIRCRAFT_DISCIPLINES

PACKAGE = Path(cdadt.__file__).resolve().parent

#: Modules that may name OpenConcept at all. The black box loads it by string, so this is empty;
#: the constant exists to make the intent explicit rather than implicit in an empty set.
MODULES_ALLOWED_TO_IMPORT_OPENCONCEPT: frozenset[str] = frozenset()


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
def test_no_cdadt_module_imports_openconcept():
    """The dependency is a string in a case file, not an import statement."""
    offenders = []
    for source in _cdadt_sources():
        module = source.relative_to(PACKAGE).with_suffix("").as_posix().replace("/", ".")
        if module in MODULES_ALLOWED_TO_IMPORT_OPENCONCEPT:
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

    assert not offenders, "cdadt must not import OpenConcept:\n" + "\n".join(offenders)


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

#: How many distinctive lines may be shared before it stops looking like coincidence. Measured:
#: the two packages currently share six, every one of them either an OpenMDAO API call or a line
#: quoted inside a cdadt docstring to say which reference function a module is the counterpart
#: of. Citing the reference is the opposite of hiding a copy of it.
MAX_SHARED_LINES = 12


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

    Compared as sets of distinctive source lines. A lifted component would show up as a run of
    dozens of them; what is actually shared is a handful of OpenMDAO API calls and lines quoted
    in cdadt docstrings to name the reference each module answers to.
    """
    shared = sorted(_distinctive_lines(PACKAGE) & _distinctive_lines(_openconcept_repository() / "openconcept"))

    assert len(shared) <= MAX_SHARED_LINES, (
        f"cdadt shares {len(shared)} distinctive source lines with OpenConcept, which looks "
        f"like copied source rather than coincidence:\n  " + "\n  ".join(shared[:20])
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
            if isinstance(node, (ast.Import, ast.ImportFrom))
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
    """The shipped box publishes the whole weight breakdown; record it if that ever changes."""
    unavailable = converged_analysis.aircraft.missing(built_box)
    assert not unavailable, "The shipped black box no longer publishes: " + "; ".join(
        f"{d}: {', '.join(names)}" for d, names in unavailable.items()
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
                    value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
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
                if isinstance(value, (dict, list, set)):
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
