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

# Cross-references to things Sphinx cannot resolve are treated as errors (the docs build
# with -W), with these exceptions:
#   - OpenConcept publishes no intersphinx inventory, so none of its classes resolve.
#   - OpenMDAO's inventory does not cover every internal class path referenced here.
#   - cdadt.tests is not autodoc'd; test modules are referenced by name so a reader knows
#     which file enforces a stated rule.
nitpick_ignore_regex = [
    ("py:.*", r"openconcept\..*"),
    ("py:.*", r"openmdao\..*"),
    ("py:.*", r"cdadt\.tests\..*"),
    # Private base classes appear in `:show-inheritance:` chains but are not documented,
    # which is the intent: they are implementation detail, not public API.
    ("py:.*", r"cdadt\..*\._[A-Za-z].*"),
]

try:
    from sphinx_mdolab_theme.config import *  # noqa: F401,F403  (matches OpenConcept's docs)
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]
