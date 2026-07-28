"""Shared helpers for the cdadt test suite.

Nothing in this module fakes physics. It only locates the installed OpenConcept clone on
disk so that tests can assert cdadt has not modified it, and reports which optimizers the
running environment actually provides.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path


def openconcept_package_dir() -> Path:
    """Return the directory containing the imported ``openconcept`` package.

    Returns
    -------
    pathlib.Path
        Directory of the ``openconcept`` package that Python actually imports. This
        resolves through the editable install, so it points at the working clone rather
        than at a site-packages copy.
    """
    module = importlib.import_module("openconcept")
    if module.__file__ is None:  # pragma: no cover - namespace package, not expected
        raise RuntimeError("openconcept was imported as a namespace package with no __file__")
    return Path(module.__file__).resolve().parent


def openconcept_repo_root() -> Path | None:
    """Return the git working tree root of the imported OpenConcept, if it is one.

    Returns
    -------
    pathlib.Path or None
        Path to the repository root containing a ``.git`` entry, or ``None`` if the
        imported OpenConcept is not inside a git working tree (for example, a plain
        wheel install from PyPI).
    """
    for candidate in [openconcept_package_dir(), *openconcept_package_dir().parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def git_status_porcelain(repo_root: Path) -> str:
    """Return ``git status --porcelain`` output for a repository.

    Parameters
    ----------
    repo_root : pathlib.Path
        Root of the git working tree to inspect.

    Returns
    -------
    str
        Raw porcelain status output. An empty string means the working tree is clean.
    """
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def available_pyoptsparse_optimizers() -> set[str]:
    """Return the set of optimizer names pyOptSparse can actually construct.

    Returns
    -------
    set of str
        Names such as ``{"IPOPT", "SLSQP"}``. Empty if pyOptSparse is not importable.
        Membership means the optimizer was successfully instantiated, not merely that the
        symbol exists.
    """
    try:
        import pyoptsparse
    except ImportError:
        return set()

    found = set()
    for name in ("IPOPT", "SNOPT", "SLSQP", "CONMIN", "PSQP", "NSGA2", "ALPSO", "ParOpt"):
        optimizer_class = getattr(pyoptsparse, name, None)
        if optimizer_class is None:
            continue
        try:
            optimizer_class()
        except Exception:
            continue
        found.add(name)
    return found
