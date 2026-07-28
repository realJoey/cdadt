"""Sphinx configuration for the cdadt documentation."""

from __future__ import annotations

import cdadt

project = "cdadt"
author = "Joey Gould"
copyright = "2026, Joey Gould"
version = cdadt.__version__
release = cdadt.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

# Napoleon rather than numpydoc, because the discipline docstrings use OpenConcept's own
# "Inputs / Outputs / Options" section names for the OpenMDAO variables a group promotes.
# Those are the right names for the reader and numpydoc has no way to accept them.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_custom_sections = [
    ("Inputs", "params_style"),
    ("Outputs", "params_style"),
    ("Options", "params_style"),
]

autodoc_member_order = "bysource"
# Members are requested explicitly per directive. Turning them on globally documents the
# re-exports in every package __init__ as well as their defining module, which Sphinx then
# reports as duplicate objects.
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "openmdao": ("https://openmdao.org/newdocs/versions/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "cdadt"
