GAMSPy 1.18.0 (2025-12-01)
==========================

Improvements in existing functionality
--------------------------------------
- #727: Accept the path of an existing solver options file as a valid "solver_options" argument in model.solve and container.writeSolverOptions.

Bug fixes
---------
- #728: Decouple container and socket connection to allow pickling of symbols and containers.
- #729: Fix name mapping of GAMSDict and GAMSDictMap file formats in model.convert function.
- #732: Fix model and path issues of "gamspy run miro" command on Windows.

Miscellaneous internal changes
------------------------------
- #731: Change expected type of the "output" argument to TextIO instead of TextIOWrapper in model.solve.

