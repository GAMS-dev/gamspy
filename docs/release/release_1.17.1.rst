GAMSPy 1.17.1 (2025-10-30)
==========================

New features
------------
- #713: Implement softplus activation function

Bug fixes
---------
- #713: Fix bug in `lse_max_sc` where order of arguments were incorrectly put.

Improved documentation
----------------------
- #713: Update docs for softplus activation function.

CI/CD changes
-------------
- #714: Add a new scheduled job to automatically upgrade pre-commit dependency versions.

Miscellaneous internal changes
------------------------------
- #715: Do not install extras of gamsapi and add a direct dependency to pandas.
- #716: Add stdcge (a standard cge model) to the test suite.

