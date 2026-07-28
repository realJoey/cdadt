"""Sphinx configuration for the cdadt documentation."""

import cdadt

project = "cdadt"
copyright = "2026, Joey Gould"
author = "Joey Gould"
version = cdadt.__version__
release = cdadt.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autosummary_generate = True
autodoc_member_order = "bysource"
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "openmdao": ("https://openmdao.org/newdocs/versions/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

try:
    from sphinx_mdolab_theme.config import *  # noqa: F401,F403  (matches OpenConcept's docs)
except ImportError:
    html_theme = "alabaster"

html_static_path = []
