GAMSPy 1.15.0 (2025-08-14)
==========================

New features
------------
- #653: Allow bulk setRecords via Container.setRecords function.
- #680: Add formulation for Leaky ReLU activation function.

Improvements in existing functionality
--------------------------------------
- #681: Allow redirecting generateGamsString output to a file.
  Generate unique gdx files for each execution if the debugging level is set to "keep".

Bug fixes
---------
- #674: Fix the bug that causes solve link option of one model to propagate to another model.

Improved documentation
----------------------
- #680: Minor docs fix and add docs fix for Leaky ReLU activation function.

Miscellaneous internal changes
------------------------------
- #450: Add a new model (tsp) to the model library.

