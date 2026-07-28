"""Contract: the cdadt/OpenConcept boundary, and the ownership map that spans it.

Four things are asserted here, each of which the rest of the project's claims depend on.

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
