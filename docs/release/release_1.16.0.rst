GAMSPy 1.16.0 (2025-09-16)
==========================

New features
------------
- #679: Now sets with single element generated using `dim` are singleton sets.

Improvements in existing functionality
--------------------------------------
- #639: Use temporary sets instead of temporary parameters to get the result of set-based expressions.
- #690: Improve model generation speed for RandomForest and GradientBoosting.
- #698: If no indices are provided in an assignment, assume that the operation is over the whole domain.
- #699: Allow the definition of a nonbinding equation without an equality operator. GAMSPy will automatically add == 0 at the end of the expression.
- #702: Use ProxyManager in case the user specifies HTTPS_PROXY or HTTP_PROXY.

Bug fixes
---------
- #683: Fix records filtering bug for subset indices.
- #684: In model.toGams call, write aliased set to the .gms file as well in case an alias is used as a domain.

Improved documentation
----------------------
- #679: Update matrix operations document for scalar extraction and style changes.
- #692: Add Traveling Saleman Problem (TSP) in Notebook examples.
- #697: Add documentation for model types required for formulations.

Dependencies
------------
- #679: Upgrade gamspy_base and gamsapi to 51.1.0.

Miscellaneous internal changes
------------------------------
- #695: .records call will return either a DataFrame or None. It will stop squeezing single rows into a float.

