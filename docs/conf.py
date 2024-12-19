# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import gamspy
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GAMSPy"
copyright = f"{datetime.now().year}, GAMS Development Corporation"
author = "GAMS"
version = gamspy.__version__
language = "en"
html_last_updated_fmt = ""  # to reveal the build date in the pages meta

# -- Switcher ----------------------------------------------------------------
# Define the json_url for our version switcher.
json_url = "https://gamspy.readthedocs.io/en/latest/_static/switcher.json"

is_readthedocs = os.environ.get("READTHEDOCS_VERSION", "dev")
if is_readthedocs == "dev":
    json_url = "docs/_build/html/_static/switcher.json"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx_design",
    "sphinx_copybutton",
    "nbsphinx",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
    "sphinx_tabs.tabs",
]

copybutton_prompt_text = ">>> "
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GAMS-dev/gamspy",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "X",
            "url": "https://twitter.com/GamsSoftware",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "GAMS",
            "url": "https://www.gams.com",
            "icon": "_static/gams.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "logo": {
        "image_light": "_static/gamspy_logo.png",
        "image_dark": "_static/gamspy_logo_dark.png",
    },
    "switcher": {
        "json_url": json_url,
        "version_match": f"v{version}",
    },
    "navbar_center": ["version-switcher", "navbar-nav"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "pygments_light_style": "tango",
    "pygments_dark_style": "lightbulb",
}

autodoc_member_order = "groupwise"

# Display todos by setting to True
todo_include_todos = False

favicons = ["gams.ico"]
