GAMSPy 0.14.1
-------------
- General

  - Add SOS1 ReLU implementation.
  - Add __repr__ to all GAMSPy language constructs for better debugging.
  - Give a warning in case the domain is not initialized by the time there is an attribute assigment.
  - Allow indexing on alias symbols.
  - Add reference_file option.
  - Add selective loading for solve statements.
  - Change default port to communicate with license server to 443.
  - Fix installing licenses from a path.

- Documentation

  - Add API docs for SOS1 ReLU implementation.
  - Explain the working directory - debugging level relationship.

- Testing

  - Add tests for SOS1 ReLU implementation.
  - Shorten attribute assignments in model library (variable.l[...] = ... -> variable.l = ...).
  - Add tests for indexing on alias symbols.
  - Test selective loading for solve statements.
  - Add new install license tests.
  - Add a new model (coex) to the model library.
