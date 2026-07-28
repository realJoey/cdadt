"""Enforcement of the single-module OpenConcept boundary.

Claim class: ``unit``. The black box is only a black box if there is exactly one place that
knows what is inside it. That property is easy to state, easy to believe, and easy to lose
one convenience import at a time -- so it is checked by scanning cdadt's own source rather
than by review.

If this fails, the fix is to add what is needed to :mod:`cdadt.mission.contract` and reach
it through :class:`~cdadt.mission.blackbox.MissionBlackBox`, not to widen the allowlist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import cdadt

pytestmark = pytest.mark.unit

MISSION_IMPORT_ALLOWLIST = {
    # The black box itself: the one module that builds FullMissionWithReserve.
    "cdadt/mission/blackbox.py",
    # Contract tests must build a real OpenConcept mission to verify the contract against.
    "cdadt/tests/mission/test_contract.py",
    # The environment gate builds a real mission to prove the toolchain works.
    "cdadt/tests/test_environment.py",
    # This file names the module it is enforcing against.
    "cdadt/tests/mission/test_boundary.py",
}


def _package_sources():
    """Yield every cdadt source file with its repository-relative POSIX path."""
    package_root = Path(cdadt.__file__).resolve().parent
    for path in sorted(package_root.rglob("*.py")):
        yield path, path.relative_to(package_root.parent).as_posix()


def _imported_openconcept_modules(path: Path) -> set[str]:
    """Return the OpenConcept submodules a source file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names if alias.name.split(".")[0] == "openconcept")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "openconcept":
            imported.add(node.module)
    return imported


def test_only_the_blackbox_imports_openconcept_mission():
    """``openconcept.mission`` is imported by the black box and its own tests, nothing else.

    Importing the mission machinery elsewhere means a second module knows the shape of the
    dependency, and replacing or upgrading it stops being a two-file change.
    """
    offenders = {}
    for path, relative in _package_sources():
        if relative in MISSION_IMPORT_ALLOWLIST:
            continue
        mission_imports = {m for m in _imported_openconcept_modules(path) if m.startswith("openconcept.mission")}
        if mission_imports:
            offenders[relative] = sorted(mission_imports)

    assert not offenders, (
        "Only cdadt/mission/blackbox.py may import openconcept.mission:\n"
        + "\n".join(f"  {path}: {modules}" for path, modules in sorted(offenders.items()))
        + "\nAdd what you need to cdadt/mission/contract.py and read it through MissionBlackBox."
    )


def test_the_blackbox_actually_imports_openconcept():
    """The allowlist is not vacuous: the black box really does import the dependency.

    Without this, deleting the import would make the previous test pass trivially.
    """
    package_root = Path(cdadt.__file__).resolve().parent
    imports = _imported_openconcept_modules(package_root / "mission" / "blackbox.py")
    assert any(
        module.startswith("openconcept.mission") for module in imports
    ), f"cdadt/mission/blackbox.py does not import openconcept.mission; found {sorted(imports)}"


def test_no_cdadt_module_outside_the_mission_package_imports_openconcept_at_all():
    """Disciplines and providers may wrap OpenConcept components, but nothing else may.

    Providers under ``cdadt/providers/openconcept/`` exist precisely to wrap OpenConcept's
    component library, so they are expected to import it. Certification, optimization, core
    and I/O must not: a regulation or an optimizer that reaches into the dependency directly
    is a boundary violation that the mission-module check above would not catch.
    """
    forbidden_prefixes = ("cdadt/core/", "cdadt/certification/", "cdadt/optimization/", "cdadt/io/")

    offenders = {}
    for path, relative in _package_sources():
        if not relative.startswith(forbidden_prefixes):
            continue
        imports = _imported_openconcept_modules(path)
        if imports:
            offenders[relative] = sorted(imports)

    assert not offenders, (
        "These packages must not import OpenConcept directly:\n"
        + "\n".join(f"  {path}: {modules}" for path, modules in sorted(offenders.items()))
        + "\nWrap it in a provider under cdadt/providers/, or read it through MissionBlackBox."
    )
