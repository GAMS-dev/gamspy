GAMSPy 1.21.0 (2026-03-06)
==========================

New features
------------
- #778: Support len operation on all symbols, and operations.
- #786: Support "loop" control structure of GAMS via gp.Loop context manager.

Improvements in existing functionality
--------------------------------------
- #773: Allow renaming the output of model.convert function.
- #775: Provide load_symbols explicitly when it's known in synch_with_gams calls.
- #779: Only unload the GDX library when the GAMS system directory changes.
- #783: Get rid of all suffixes if the input file to 'gamspy mps2gms' is a zip file (e.g. indus89.mps.gz). The output python file will be named indus89.py instead of indus89.mps.py.
- #785: Use model attributes to return the summary of the solve instead of parsing the trace file.

Bug fixes
---------
- #784: Mitigate the issue of one environment affecting another on the same machine with package managers that keep a global cache (e.g. pixi).

