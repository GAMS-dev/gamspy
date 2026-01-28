GAMSPy 1.19.0 (2026-01-27)
==========================

Improvements in existing functionality
--------------------------------------
- #763: Accept Path-like load_from argument in container.loadRecordsFromGdx function.

Bug fixes
---------
- #763: Make loadRecordsFromGdx respect .synchronize attribute of symbols.

Improved documentation
----------------------
- #752: Add more examples in API reference.

CI/CD changes
-------------
- #757: Replace pre-commit with prek.

Miscellaneous internal changes
------------------------------
- #725: Add a new example showing the usage of solution pools.
- #751: Add AGENTS.md file to guide coding agents.
- #756: Move dev, test and doc dependencies to depedency groups instead of optional dependencies.
- #758: Update the validation error message that is raised in case debugging_level is not set.

