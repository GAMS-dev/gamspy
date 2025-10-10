GAMSPy 1.17.0 (2025-10-10)
==========================

Improvements in existing functionality
--------------------------------------
- #701: Show licensed solvers in the output of "gamspy show license"
- #703: Start using griffe to ensure backward compatibility.

Bug fixes
---------
- #668: Fix synchronization bug in expert sync mode when there are other symbols defined after setting the synchronize flag.

Improved documentation
----------------------
- #710: Add Google Collab links into examples and reorder examples for better flow.

Dependencies
------------
- #707: Replace urllib3 with requests for network calls.

Miscellaneous internal changes
------------------------------
- #566: Add indus89 to the model library.
- #709: Change the default value of 'USE_PY_VAR_NAME' to 'yes-or-autogenerate'.

