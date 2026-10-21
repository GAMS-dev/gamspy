# Project Overview

GAMSPy is a fast solver-agnostic mathematical optimization package.

## Folder Structure
- `src/`: Source code
  - `gamspy/`: GAMSPy library source code.
  - `gamspy_cli/`: GAMSPy CLI source code.
- `tests/`: Test suite.
- `docs/`: Sphinx documentation. 
- `scripts/`: Development, build, and performance scripts.
- `ci/`: Yaml files for Gitlab CI jobs.
- `changelog/`: Changelog fragments to be used by towncrier.

## Core Architecture (`src/gamspy` folder)
- `_symbols/` - Implementation of the main symbols such as Set, Alias, UniverseAlias, Parameter, Variable, and Equation.
- `_backend/` - Implementation of different backends such as local backend, GAMS Engine and NEOS Server to solve models.
- `_execution` - Different execution engines for the local backend.
- `_algebra/` - Implementation of other structures to help developing models such as Expression, Domain, Number, Sum, Product, Smin, and Smax.
- `formulations/` - Implementation of mathematical formulations including machine learning and neural network formulations.
- `math` - Math functions.
- `_container.py` - Implementation of the Container.
- `_model.py` - Implementation of the model.
- `_records_ingestion.py` - Implementation of the ingress of user records.
- `_types.py` - GAMSPy types.

## Setting Up A Development Environment
- Run `uv sync --group dev --group test --group doc` to get the necessary environment for development.
- If a virtual environment is not already activated, run `source .venv/bin/activate` to activate the environment.
- Run `uv pip install -e .` before you do any changes.
- Install a solver with `gamspy install solver <solver_name>` when necessary.

## Testing
- You can find all available markers in `pyproject.toml`.
- Run `pytest -m 'unit'` to run all unit tests.
- Run `pytest -m 'integration'` to run all integration tests.
- Run `pytest -m 'model_library'` to run all models in the model library.
- Run `pytest -m 'doc'` to run documentation tests.
- Run `pytest -m 'cli'` to run cli tests.

## Documentation
- Run `cd docs && make html` to build the documentation.
- Run `cd docs && make linkchecker` to check the validitiy of the links in the documentation.

## Validating Changes
- Always run `prek run --all-files` before declaring any work complete, then fix all errors before moving forward.
- Always run `pytest -m 'unit or integration'` before declaring any work complete, then fix all errors before moving forward.
- `pytest -m 'model_library'` is the final gate for any source code changes. This requires a GAMSPy license and all solvers. If a GAMSPy license is not available, do not run model_library tests.
- `pytest -m 'cli'` is the final gate for any cli changes.


## Coding Guidelines
- Follow ruff rules specified in pyproject.toml for formatting and linting. 


### Code Quality

- Prefer small, focused changes. Avoid unrelated refactoring and unnecessary changes to public APIs.
- Add appropriate type annotations to new or modified functions and variables where they improve clarity and conform to existing project conventions. Avoid introducing inaccurate or unnecessarily complex types. Avoid using Any as a type unless it's absolutely necessary.
- Reuse and extend existing code whenever possible.
