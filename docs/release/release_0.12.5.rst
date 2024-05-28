GAMSPy 0.12.5
=============

- General

  - Do not pick the default solver if the given solver is not compatible with the problem type.
  - Add extrinsic function support.
  - Expose addGamsCode to user.
  - Refactor the underlying implementation of options.
  - Show better error messages.
  - Fix number of arguments that log_gamma takes.
  - Rename getStatement to getDeclaration.

- Testing

  - Add tests for extrinsic functions.
  - Test whether the given solver is capable of solving the problem type.
  - Add an addGamsCode test for each problem type. 
  - Test Jupyter Notebooks in docs automatically.
  - update log option tests.

- Documentation

  - Remove unnecessary GTP functions from documentation
  - Add a doctest for addGamsCode.
  - Update the documentation on generating log files.
  