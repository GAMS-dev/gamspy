GAMSPy 1.5.1
------------

Release Date: 31.01.2025

- General

  - Fix the bugs in dynamic set assignments.
  - Rewrite parts of GAMS Control API.
  - Fix debugging level bug of NEOS backend.
  - Fix license issue of big models that are solved with frozen solve.
  - Allow loadRecordsFromGdx to domain forward.
  - Enhance bound propagation for `MaxPool2d` and `MinPool2d` classes.

- Testing

  - Add bound propagation tests for `MaxPool2d` and `MinPool2d` classes.

- Documentation

  - Update embedding Neural Network documentation.