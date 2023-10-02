# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GAMSpy"
copyright = "2023, GAMS Development Corporation"
author = "GAMS"
release = "0.9.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.autosummary",
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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GAMS-dev",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/GamsSoftware",
            "icon": "fa-brands fa-square-twitter",
        },
    ],
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "show_nav_level": 2,
}
