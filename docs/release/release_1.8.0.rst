GAMSPy 1.8.0
------------

Release Date: 16.03.2025

- General

  - Improve the performance of frozen solves.
  - Add support for new equation, variable matching syntax for MCP models.
  - Ignore empty and newlines in the existing solvers file.
  - Use finalizers instead of __del__.
  - Cache solver capabilities, default solvers and installed solvers to speed up solver validation.
  - Fix the bug in the case of multiple frozen models in one container.
  - Perform pip audit check in the pipeline instead of pre-commit.

- Documentation

  - Add `Examples` section under `Machine Learning` documentation.
  - Add a Thermal Reformer example demonstrating neural network surrogate modeling.

- Testing

  - Fix the issue of mac jobs deleting each others environments.
