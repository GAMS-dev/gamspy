GAMSPy 0.13.5
=============

- General
  
  - Make trace file name dynamic to avoid race condition on parallel runs.
  - Fix log options for GAMS Engine backend.
  - Initial support for GAMSPy to Latex.
  - Generate solver options file under container working directory instead of current directory.
  - Fix implicit set issues for toGams function.

- Documentation

  - Add links to the api reference for symbols and functions mentioned in the documentation.
  - Minor documentation corrections.

- Testing

  - Logout from GAMS Engine only on Python 3.12 to avoid unauthorized calls on parallel jobs.
  - Add tests to verify the behaviour of different logoption values.
  - Add tests for GAMSPy to Latex.
  