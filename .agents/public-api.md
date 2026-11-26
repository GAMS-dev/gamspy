# Public API

## What counts as public

A name is public if it can be imported through a path with no `_`-prefixed component, for example `gamspy.Set`, `gamspy.math.exp` or `gamspy.formulations.pwl_interval_formulation`. Re-exports in `gamspy/__init__.py` count. Code under `gamspy._model` is private, but `gamspy.Model` is public.

## Backward compatibility

- Each release runs `griffe check` against the previous tag (`scripts/update_version.py`). Removing or renaming a public name or parameter, or changing a default, fails the release.
- To remove or change something: keep it working, emit a deprecation warning, and add a `deprecation` changelog fragment.

## Adding a public name

1. Export it and add it to the relevant `__all__` (`gamspy/__init__.py`, `gamspy/math/__init__.py`, `gamspy/formulations/__init__.py`, ...).
2. Give it a docstring with a runnable example (see [python-conventions](python-conventions.md)).
3. Reference docs: submodules (`gamspy.formulations`, `gamspy.math`, `gamspy.utils`, `gamspy.exceptions`) use `automodule`, which picks up `__all__` automatically. A new top-level `gamspy.X` needs its own page like `docs/reference/gamspy.Sum.rst` and an entry in the toctree in `docs/reference/gamspy.rst`. User-facing features also get a section in the matching `docs/user/` guide (see [docs](docs.md)).
4. Use a `feature` changelog fragment.
