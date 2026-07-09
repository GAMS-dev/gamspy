GAMSPy 1.26.1 (2026-07-09)
==========================

Improvements in existing functionality
--------------------------------------
- #860: Improve the performance of loop control structures by employing the new newOrChangedNoData option in GAMS.

Bug fixes
---------
- #860: Fixes a false domain violation when indexing a symbol with a nested implicit set that spans multiple positions. The nested set's positions are now expanded correctly so subsequent indices stay aligned with the declared domain.


