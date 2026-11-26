# Git Workflow

## Branches

- Branch from `develop`, named `<issue_number>-<slug>` (e.g. `981-enhance-pwl-interval-formulation`).
- Merge requests go to `develop`.

## Changelog fragments

**Every change needs a fragment**, including internal ones.

- Path: `changelog/<issue>.<type>.rst`. `<issue_number>` is the issue number at the start of the branch name.
- `<type>` is one of `feature`, `improvement`, `bugfix`, `doc`, `deprecation`, `dependency`, `cicd`, `misc` (defined in `pyproject.toml`, `[[tool.towncrier.type]]`).
- Content: one short reStructuredText paragraph describing the change from the user's point of view. Only `feature` entries may run to several paragraphs.
- Don't edit `CHANGELOG.rst`; towncrier builds it at release time.

## Release-owned files

`scripts/update_version.py` updates these during a release. Don't change them in normal work:

- `version` in `pyproject.toml` and `tests/test_gamspy.py::test_version`
- `docs/_static/switcher.json`, `docs/release/`, `CHANGELOG.rst`
