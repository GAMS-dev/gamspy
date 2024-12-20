GAMSPy 1.4.0
------------

Release Date: 20.12.2024

- General

  - Resolve static code analysis issues to improve code quality.
  - Return the value as a float if the given domain sets are all literals.
  - Add an automation script to update pyproject.toml, switcher, version test, and the release notes.
  - Allow propagating bounds of the output in the Linear class.

- Testing

  - Set COVERAGE_CORE to sysmon to make use of the new sys.monitoring package in Python.

- Documentation

  - Add an example demonstrating how to solve the Minimum Cost Multi-Commodity Flow Problem using Column Generation in GAMSPy.
  - Remove non-negative variable type from the docs.
  - Add plausible.js for analytics.
  - Minor update in embedding nn documentation.
  