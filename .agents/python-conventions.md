# Python Conventions

prek enforces ruff and `ty`; `pyproject.toml` has the full rule set. The rules below either aren't enforced automatically or are the ones that most often fail.

## Typing

- Every new or changed function in `src/` needs full annotations, including the return type (ruff `ANN`; tests are exempt).
- Avoid `Any`. A `# type: ignore` must name the rule (`PGH003`). Put typing-only imports under `if TYPE_CHECKING:` (`TC`).
- Every non-init module in `src/gamspy` starts with `from __future__ import annotations`; keep that in new modules.

## Docstrings and doctests

- Public objects use numpydoc docstrings with an `Examples` section.
- `pytest -m doc` runs doctest over **every file in `src/`**, one `test_docs[<path>]` test per file, so examples must actually run.
- Add `# doctest: +SKIP` to example lines that need GAMS Engine, NEOS, a full license or an optional dependency (`torch`, `graphviz`).

## Errors

- Invalid user input raises `gamspy.exceptions.ValidationError`.
- `GamspyException` and its subclasses are for errors that GAMS itself reports. Don't raise them for bad input.

## Reuse

- Check `gamspy.utils`, `_validation.py` and `formulations/utils.py` before writing a new helper.
