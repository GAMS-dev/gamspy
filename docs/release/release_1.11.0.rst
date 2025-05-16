GAMSPy 1.11.0 (2025-05-16)
==========================

New features
============

- #607: Allow evaluation of expression on the fly without requiring an explicit assignment statement to a parameter.


Bug fixes
=========

- #608: Fix literal bug in latex representation of implicit symbols.

- #633: Incrementally build model declaration to avoid input line length limitation (80000 characters).

- #638: Fix the bug that occurs when "gamspy probe -h" runs.


Improved documentation
======================

- #614: Add developer guide to the documentation.


CI/CD changes
=============

- #631: Add tests for Linux arm64. Add a new marker called "requires_license" to separate tests that require a license to run.


