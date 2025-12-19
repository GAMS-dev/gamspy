GAMSPy 1.18.3 (2025-12-19)
==========================

New features
------------
- #712: Integrate mps2gms to gamspy cli.

Improvements in existing functionality
--------------------------------------
- #730: Read the values of most model attributes from the trace file. Lazily read others if requested.
- #740: Improve the alignment of constraints in the result of model.toLatex call.

Bug fixes
---------
- #742: Do not perform definition valiation on MPSGE equations since they are generated on the GAMS side.
- #743: Use gp.Number instead of a Python variable in nonbinding equations of MCP models.
- #745: Retrieve universe alias name from stack in case it's not provided instead of autogenerating a name.

