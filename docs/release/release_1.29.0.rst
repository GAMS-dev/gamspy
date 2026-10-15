GAMSPy 1.29.0 (2026-10-15)
==========================

New features
------------
- #912: Added ``minimum``, ``maximum`` and ``absolute`` formulations to ``gamspy.formulations`` for modeling minimum, maximum and absolute value constraints.

Bug fixes
---------
- #912: Fix ``indicator`` failing when ``USE_PY_VAR_NAME`` is ``"yes"`` and adding symbols named after its internal Python variables otherwise.
- #921: Fix incorrect equation type bug on repeated GDX writes.
- #922: Fix writing and loading GDX files whose path contains a space, and resolve relative GDX paths against the current working directory of Python.
- #926: Fix wrong values being sent to GAMS when the records of a parameter are a column sliced from the records of a variable or equation.
- #928: Fix incorrect and invalid LaTeX output of latexRepr and toLatex.
- #929: Fix xor precedence.
- #932: Fixed the expiration date of time-limited licenses in ``gamspy show license``, which was 30 days earlier than the date GAMS reports.

Improved documentation
----------------------
- #918: Fix swapped wet and dry season inflow data in the ClearLake SDDP example and update the worked examples that the documentation derives from it.

Dependencies
------------
- #927: Add Python 3.15 support and drop Python 3.10 support.

Miscellaneous internal changes
------------------------------
- #893: Add linopy to IJKLM benchmark.
- #930: Unify the declaration, redeclaration, serialization and records handling of ``Set``, ``Parameter``, ``Variable`` and ``Equation`` in their common base class.
- #931: Set threads to 1 by default in model library tests to avoid cpu oversubscription in pipelines.


