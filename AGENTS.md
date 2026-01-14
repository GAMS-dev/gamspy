## Project Overview

GAMSPy is built with Python. The codebase is organized into the following layers:

### Root Folders
- `src/`: Source code
- `tests/`: Unit and integration tests.
- `docs/`: Sphinx documentation. 
- `scripts/`: Development, build, and performance scripts.
- `ci/`: Yaml files for Gitlab CI jobs.

### Core Architecture (`src/gamspy` folder)
- `_symbols/` - Implementation of the main symbols in GAMSPy such as Set, Parameter, Variable, Equation and Alias.
- `_backend/` - Implementation of different backends such as GAMS Engine and NEOS Server to solve models.
- `_cli/` - GAMSPy CLI implementation.
- `_algebra/` - Implementation of other structures to help developing models such as Expression, Domain, Number, Sum, Product, Smin, and Smax.
- `formulations/` - Implementation of different formulations including machine learning and neural network formulations.
- `math` - Math related expressions.
- `_container.py` - Implementation of the Container.
- `_model.py` - Implementation of the model.

## Setting Up A Development Environment
- Run `uv sync --group dev --group test --group doc` to get the necessary environment for development.
- If a virtual environment is not already activated, run `source .venv/bin/activate` to activate the environment.
- Run `uv pip install -e .` before you do any changes.

## Testing
- You can find all available markers in `pyproject.toml`.
- Run `pytest -m 'unit'` to run all unit tests.
- Run `pytest -m 'integration'` to run all integration tests.
- Run `pytest -m 'model_library'` to run all models in the model library. This requires a GAMS license. Install GAMS license with `gamspy install license $ACCESS_CODE` if `ACCESS_CODE` is available in the environment variables or if it exists in the `.env` file. If `ACCESS_CODE` is not available, do not run model_library tests.
- Run `pytest -m 'doc'` to run documentation tests.

## Documentation
- Run `cd docs && make html` to build the documentation.
- Run `cd docs && make linkchecker` to check the validitiy of the links in the documentation.

## Validating Changes
- Always run `prek run --all-files` before declaring any work complete, then fix all errors before moving forward.
- Always run `pytest -m 'unit or integration'` before declaring any work complete, then fix all errors before moving forward.


## Coding Guidelines
- Follow ruff rules specified in pyproject.toml for formatting and linting. 


### Code Quality

- If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task
- Never duplicate imports. Always reuse existing imports if they are present.
- Always use type annotations.
- Do not duplicate code. Always look for existing utility functions, helpers, or patterns in the codebase before implementing new functionality. Reuse and extend existing code whenever possible.
