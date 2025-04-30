GAMSPy 1.10.0 (2025-04-30)
==========================

New features
============

- #599: Allow renaming on ``container.loadRecordsFromGdx`` function call.

- #601: Allow users to disable all validation via ``gp.set_options({"VALIDATION": 0})``.


Improvements in existing functionality
======================================

- #594: Add GAMSPyJacobian file format for the convert function.

- #612: Raise a validation error in case an automatically named symbol is used in an equation of an EMP model. Use base64 auto-generated names instead of plain uuid.uuid4 names.

- #613: Improve the error message of undefined scalar equations.

- #615: Remove duplicate conopt entries in gamspy list solvers cli command and add utils.getInstallableSolvers function.

- #617: Cast the type of objective value, num equations, num variables, and solver time in the summary dataframe.

- #623: Improve the error message in case the user does not have an internet connection or the PyPI server are down.

- #624: Allow .records call on implicit variables and equations.


Bug fixes
=========

- #625: Overload __eq__ and __ne__ magic functions of the Number class to ensure the order is correct in expressions.

- #626: Fix the bug in the filtering of a single record in non-level attributes of a variable (lo, up, marginal, scale).

- #629: Allow record filtering over eq.range, eq.slacklo, eq.slackup, eq.slack, and eq.infeas attributes.


Improved documentation
======================

- #602: Use towncrier to automate changelog creation and avoid marge conflicts in the changelog file.


