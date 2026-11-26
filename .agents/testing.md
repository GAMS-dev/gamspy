# Testing

## Commands

| Goal | Command |
|---|---|
| Fast check | `pytest tests/unit/test_set.py -k "test_name" -x` |
| Default gate | `pytest -m "unit or integration" -n auto` |
| Default gate without a license | `pytest -m "(unit or integration) and not requires_license" -n auto` |
| Docstring and doc-consistency checks | `pytest -m doc -n auto` |
| One file's doctests | `pytest -m doc -k "gamspy/_model.py"` |
| CLI | `pytest -m cli` |
| Install a missing solver | `gamspy install solver <name>` |

## Extra gates by area

- You changed docstrings, public APIs or options: run `doc`.
- You changed `src/gamspy_cli/`: run `cli`.
- You changed `src/gamspy/`: `model_library` is the final gate. It needs a non-demo license and all solvers; if they're missing, skip it and say so.

## Markers

**Every test needs a marker**, either `pytestmark = pytest.mark.unit` at module level or `@pytest.mark.unit` on the test. CI selects tests by marker, so an unmarked test never runs in CI.

| Marker | Use for |
|---|---|
| `unit` | Files in `tests/unit/`: behavior of one symbol, function or formulation. Small solves are fine. The default for new tests. |
| `integration` | Files in `tests/integration/`: whole workflows, such as solve behavior across solvers and statuses, frozen models, external modules, GAMS and LaTeX export, MIRO. |
| `cli` | `gamspy` cli tests (`tests/integration/test_cli.py`). |
| `doc` | Doctests and checks that docs match code (`tests/test_gamspy.py`). |
| `model_library` | `tests/integration/models/*.py`; each file is a standalone script run as a subprocess. |
| `requires_license` | Fails under the demo license. Add it alongside `unit`/`integration`. |
| `requires_solvers("cplex", ...)` | Skipped automatically when a solver is missing. |
| `engine`, `neos` | Need credentials from `.env`. Don't run these unless asked. |

## Conventions

- Add a new test to the existing file for the same feature and use that file's marker.
- Use the fixtures from `tests/conftest.py` if possible.
- Don't remove files from `container.working_directory` manually; the container deletes the directory when it goes out of scope.
