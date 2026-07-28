"""Guard that cdadt never modifies the OpenConcept dependency.

Claim class: ``unit``. OpenConcept is a read-only external dependency: cdadt imports it and
must never edit, vendor, or monkey-patch it. That rule is easy to state and easy to break
by accident, so it is enforced by the test suite rather than by review.
"""

from __future__ import annotations

import pytest

from cdadt.tests.support import git_status_porcelain, openconcept_repo_root

pytestmark = pytest.mark.unit


def test_openconcept_working_tree_is_clean():
    """The OpenConcept clone cdadt imports has no uncommitted modifications.

    A dirty tree means either cdadt edited OpenConcept (a rule violation) or the user has
    work in progress there. Either way the reference-validation results in
    ``docs/validation.rst`` would no longer be comparable against upstream OpenConcept, so
    this fails loudly rather than warning.

    Untracked build artifacts produced by installing OpenConcept editable
    (``openconcept.egg-info``, ``.pytest_cache``, ``__pycache__``) are not modifications
    and are excluded.
    """
    repo_root = openconcept_repo_root()
    if repo_root is None:
        pytest.fail(
            "openconcept is not installed from a git working tree, so its integrity cannot be "
            "verified. See docs/install.rst."
        )

    ignorable_suffixes = ("egg-info/", ".pytest_cache/", "__pycache__/", ".coverage")
    offending = []
    for line in git_status_porcelain(repo_root).splitlines():
        status, _, path = line.partition(" ")
        path = path.strip().strip('"')
        if status == "??" and path.endswith(ignorable_suffixes):
            continue
        offending.append(line)

    assert not offending, (
        f"The OpenConcept clone at {repo_root} has modifications:\n"
        + "\n".join(offending)
        + "\n\ncdadt must never modify OpenConcept. Revert these changes and express the "
        "difference in cdadt instead (subclass, wrap, or add a cdadt component)."
    )


def test_cdadt_does_not_monkey_patch_openconcept():
    """No cdadt module assigns attributes onto an OpenConcept module or class.

    Editing OpenConcept on disk is one way to break the read-only rule; rebinding its
    attributes at import time is the other. This scans cdadt's own source for assignments
    whose target is rooted at an ``openconcept`` import.
    """
    import ast
    from pathlib import Path

    import cdadt

    package_root = Path(cdadt.__file__).resolve().parent
    violations = []

    for source_path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

        openconcept_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "openconcept":
                        openconcept_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "openconcept":
                    for alias in node.names:
                        openconcept_names.add(alias.asname or alias.name)

        if not openconcept_names:
            continue

        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]

            for target in targets:
                root = target
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in openconcept_names:
                    violations.append(
                        f"{source_path.relative_to(package_root.parent)}:{node.lineno} "
                        f"assigns onto OpenConcept symbol '{root.id}'"
                    )

    assert not violations, "cdadt must not monkey-patch OpenConcept:\n" + "\n".join(violations)
