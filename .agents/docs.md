# Documentation

`docs/` is the Sphinx source (reStructuredText, numpydoc, nbsphinx). It is not MyST, so don't add `.md` pages.

## Commands

Building needs `make` and `pandoc` on PATH.

| Goal | Command |
|---|---|
| Build HTML (output in `docs/_build/html`) | `cd docs && make html` |
| Check links (as CI does) | `cd docs && make linkcheck` |
| Execute the notebooks (as CI does) | `pytest --nbmake docs/user/notebooks/*.ipynb` |

## Where things go

- `docs/reference/`: API reference pages. Submodule pages use `automodule`; top-level names have one page each (see [public-api](public-api.md)).
- `docs/user/`: guides. ML formulations are in `ml/`, SDDP in `sddp/`, general and piecewise formulations in `advanced/formulations.rst`, and options in `advanced/performance.rst`.
- `docs/release/` and `docs/_static/switcher.json`: release-owned; don't edit them (see [git-workflow](git-workflow.md)).
