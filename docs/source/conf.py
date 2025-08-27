import os
import sys
from datetime import date

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

project = "Simclstr"
author = "Sesdyn Geeks"
current_year = str(date.today().year)
copyright = f"{current_year}, {author}"

release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

typehints_defaults = "comma"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


autodoc_mock_imports = [
    "numpy",
    "scipy", 
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.cluster",
    "scipy.cluster.hierarchy", 
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "xlsxwriter",
    "numba",
    "pysd",
]

#Subcontent function display
toc_object_entries = False

# Do not execute notebooks during docs build. We only render them.
nbsphinx_execute = "never"

# Intersphinx mappings for external libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
}

# Napoleon settings for better Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# TODO extension settings
todo_include_todos = True