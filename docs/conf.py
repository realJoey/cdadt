"""Sphinx configuration for the cdadt documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdadt import __version__

# -- Project ----------------------------------------------------------------------------------

project = "cdadt"
author = "Joey Gould"
copyright = "2026, Joey Gould"  # Sphinx requires this name, shadowing the builtin
release = __version__
version = __version__

# -- General ----------------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Every docstring in cdadt is numpydoc-style; napoleon reads it without numpydoc's stricter
# validation, which would fail the build on the deliberate prose sections.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Class attributes documented with ``#:`` comments are part of the interface of a discipline or
# a requirement, so they belong in the rendered page next to the methods.
autodoc_class_signature = "mixed"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "openmdao": ("https://openmdao.org/newdocs/versions/latest", None),
}
# Documentation must build without network access; a missing inventory is not a build failure.
intersphinx_disabled_reftypes = ["*"]
intersphinx_timeout = 5

# -- HTML -------------------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_title = f"cdadt {release}"
