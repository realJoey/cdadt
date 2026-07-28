"""Tests that OpenConcept is used, and not altered.

"Black box" is a claim about behaviour, and behaviour claims need enforcement or they decay.
Three things are checked here, and they are the three things that would actually go wrong:

1. The OpenConcept clone in use has no uncommitted modifications. Editing a dependency to make
   one's own model work is the failure mode this repository most needs to rule out, and a
   ``git status`` is the only check that catches it.
2. No OpenConcept module cdadt actually imports diverges from upstream. A clean working tree
   says nothing about local *commits*, and this clone has three of them.
3. cdadt defines no subclass of an OpenConcept class. Subclassing is how a wrapper quietly
   becomes a fork: an overridden ``compute`` looks like composition in the import graph and is
   a modification in fact. cdadt composes OpenConcept components inside its own
   :class:`openmdao.api.Group` subclasses instead, which this test permits and distinguishes.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path

import openconcept
import pytest

import cdadt

pytestmark = pytest.mark.contract


def _cdadt_classes():
    """Yield every class defined in the cdadt package, with the module that defines it."""
    package = Path(cdadt.__file__).parent
    for module_info in pkgutil.walk_packages([str(package)], prefix="cdadt."):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_info.name:
                yield module_info.name, obj


def test_cdadt_subclasses_no_openconcept_class():
    """Composition is allowed; inheritance from OpenConcept is not.

    Overriding a method on an OpenConcept component would change its behaviour while leaving
    its source untouched, which is a modification that a diff of the clone would never show.
    """
    offenders = []
    for module_name, obj in _cdadt_classes():
        for base in obj.__mro__[1:]:
            root = base.__module__.split(".")[0]
            if root == "openconcept":
                offenders.append(f"{module_name}.{obj.__name__} inherits from {base.__module__}.{base.__name__}")
    assert not offenders, "cdadt must compose OpenConcept, never subclass it:\n" + "\n".join(offenders)


def test_the_openconcept_clone_has_no_uncommitted_changes():
    """The dependency must be the published dependency.

    Skipped rather than failed when OpenConcept is installed from a wheel rather than a clone:
    there is then nothing to modify, which is the same guarantee by other means.
    """
    root = Path(openconcept.__file__).resolve().parent.parent
    if not (root / ".git").exists():
        pytest.skip(f"OpenConcept at {root} is not a git clone; nothing to diff")

    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip() and not line.startswith("??")]
    assert not dirty, "The OpenConcept clone has uncommitted modifications:\n" + "\n".join(dirty)


def test_no_openconcept_module_cdadt_uses_diverges_from_upstream():
    """A clean working tree is not enough if the clone carries local commits.

    This clone does carry some: three commits to the OpenAeroStruct-backed aerostructural
    module, for OAS 2.x and NumPy 2 compatibility. cdadt does not import that module, and this
    test is what keeps that true -- it fails the moment a locally-patched OpenConcept file
    becomes one cdadt depends on.
    """
    root = Path(openconcept.__file__).resolve().parent.parent
    if not (root / ".git").exists():
        pytest.skip(f"OpenConcept at {root} is not a git clone; nothing to diff")

    upstream = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True,
        text=True,
    )
    if upstream.returncode != 0:
        pytest.skip("No origin/main to compare against")

    patched = {
        line.removeprefix("openconcept/").removesuffix(".py").replace("/", ".")
        for line in upstream.stdout.splitlines()
        if line.startswith("openconcept/") and line.endswith(".py")
    }
    if not patched:
        return

    # Import everything, then ask which OpenConcept modules are actually loaded.
    for module_name, _ in _cdadt_classes():
        importlib.import_module(module_name)
    loaded = {name.removeprefix("openconcept.") for name in list(sys.modules) if name.startswith("openconcept.")}

    used_and_patched = sorted(patched & loaded)
    assert not used_and_patched, (
        "cdadt depends on OpenConcept modules that this clone has patched locally:\n"
        + "\n".join(f"  openconcept/{name.replace('.', '/')}.py" for name in used_and_patched)
        + "\nEither drop the patch or stop depending on the module."
    )


def test_the_mission_comes_from_openconcept_unmodified():
    """The mission group cdadt builds is OpenConcept's own class, not a copy of it."""
    from openconcept.mission import FullMissionWithReserve

    from cdadt.model import SizingModel

    source = inspect.getsource(SizingModel.setup)
    assert "FullMissionWithReserve" in source
    assert FullMissionWithReserve.__module__.startswith("openconcept.")
