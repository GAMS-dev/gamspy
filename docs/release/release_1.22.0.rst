GAMSPy 1.22.0 (2026-04-09)
==========================

New features
------------
- #770: Add Recurrent Neural Networks (RNN) formulations.
- #787: Support "if" control structure of GAMS via gp.If context manager. Support "Break" and "Continue" via functions in gp.Loop.

Improvements in existing functionality
--------------------------------------
- #782: Import pandas and numpy lazily.

Bug fixes
---------
- #792: UniverseAlias.records call now returns all previously registered elements in the universe instead of the labels of all the data in the container.

Improved documentation
----------------------
- #789: Update the installation page for on-prem license server license installation.

Miscellaneous internal changes
------------------------------
- #770: Restructured neural network tests.


