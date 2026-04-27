GAMSPy 1.23.0 (2026-04-27)
==========================

New features
------------
- #795: Add support for RNN for TorchSequential.
- #796: Add Gated Recurrent Units (GRU) formulation.
  Add support for GRU for TorchSequential.
- #797: Support GAMS for statements via a context manager. The syntax is as follows: with gp.For(index, start, end, step, direction): ...
- #799: Implement projection and aggregation functions in gamspy.math module.

Improvements in existing functionality
--------------------------------------
- #722: Reset symbol records to its default values in case symbol.setRecords(None) is called.
- #796: Add dunder str method for all Neural network formulations.

Improved documentation
----------------------
- #798: Add a note in the documentation about the requirement of CMU serif font to generate a pdf file with model.toLatex function.

CI/CD changes
-------------
- #791: Replace pip-audit with uv audit.
- #801: Use uv trixie slim images to avoid pulling bigger images than necessary.

Miscellaneous internal changes
------------------------------
- #795: Add tests for RNN in TorchSequential formulations.
- #796: Add new tests for GRU.
  Add tests for GRU in TorchSequential formulations.


