GAMSPy is a solver-agnostic algebraic modeling package that turns Python objects into GAMS code and runs it through GAMS.

# Tooling

- Project and package manager: **uv**. Set up with `uv sync --group dev --group test --group doc` and activate `.venv`. Don't use `uv run`; it re-syncs the venv to the lockfile, which can break solvers installed with `gamspy install solver`.
- Pre-commit hooks runner: **prek**. Run with `prek run --all-files`.
- Testing framework: **pytest**. Tests are selected by marker: `pytest -m "unit or integration" -n auto`.

# Definition of Done

1. `prek run --all-files` passes.
2. `pytest -m "unit or integration" -n auto` passes. Without a GAMSPy license (`gamspy show license`), run `-m "(unit or integration) and not requires_license"` instead and say so.
3. Run the extra gates for the area you touched (doc, cli, model_library); see [testing](.agents/testing.md).
4. Every change has a changelog fragment; see [git workflow](.agents/git-workflow.md).
5. Keep the diff to what the task needs. Fix problems at their cause rather than where they show up, even if that spans several modules, but ask first if the fix reaches code the task didn't mention. Report anything else you notice in your summary instead of fixing it, including failures that also happen on a clean `develop`.

# Read before you act

- Writing or changing code in `src/` → [Python conventions](.agents/python-conventions.md)
- Adding, renaming or removing a name importable without a `_` in its path → [Public API](.agents/public-api.md)
- Writing, marking or running tests → [Testing](.agents/testing.md)
- Editing `docs/` or a docstring example → [Documentation](.agents/docs.md)
- Finding where something lives, or debugging the GAMS code a statement produced → [Architecture](.agents/architecture.md)
- Creating a branch or adding the changelog fragment → [Git workflow](.agents/git-workflow.md)
