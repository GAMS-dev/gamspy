GAMSPy 1.24.0 (2026-06-17)
==========================

New features
------------
- #815: Allow providing a regex to exclude symbol comparisons in gamspy gdx diff command.
- #842: Support GAMS runtime ElseIf and Else blocks via context managers.
- #843: Support while statement of GAMS via gp.While context manager.

Improvements in existing functionality
--------------------------------------
- #811: Improve the error message in case the container is not provided in model creation.
- #822: Improve the error message in case the file does not exist in the provided path for the license file.
- #825: Load records lazily after assignments and solve statements.
- #826: Load records that come from NEOS backend lazily.
- #827: Load records that come from GAMS Engine backend lazily.
- #828: Do not immediately load records from GAMS in case the container is created with a GDX file. Load the records only when the user asks for the records.
- #829: If the installed license is a free personal license, automatically install highs, ipopt, miles, shot and reshop.

Bug fixes
---------
- #809: Show PATHNLP in the licensed solvers list in **gamspy show license** command if the license allows using PATHNLP solver.
- #819: Fixed ``gp.deserialize`` leaving the implicit attributes of deserialized ``Variable`` and ``Equation`` symbols with an empty domain.
- #824: Fix LaTeX representation of literals in symbol indices.
- #833: Fix missing paranthesis in "~symbol" operations in chained expressions.
- #839: Fix special value representation in gp.Number.

Deprecations
------------
- #825: Remove UEL, domain violation and duplicate records related functions from container.

Miscellaneous internal changes
------------------------------
- #812: Improve the typing of the 'records' argument of symbols.
- #814: Skip equation definition checks in case container.addGamsCode is used.


