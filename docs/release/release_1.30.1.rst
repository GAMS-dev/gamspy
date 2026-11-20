GAMSPy 1.30.1 (2026-11-20)
==========================

New features
------------
- #963: Added domain info to GAMS MIRO contract.
- #966: Add ``Model.feasibilityRelaxation`` to relax equations with slack variables and find the minimal violation (sum, sum of squares or count) of an infeasible model.
- #967: Symbol names now autocomplete in container["..."] in IPython and Jupyter.

Improvements in existing functionality
--------------------------------------
- #981: Enhance the PWL Interval formulation by reducing the number of constraints it
  generates by using nonnegative distances from segment starts.

Bug fixes
---------
- #966: Fix equation definitions with ``where`` on the equation itself, e.g. ``e.where[c > 0] = x >= 2``: they are now stored as the definition of the equation, and indexed equations get their domain in the generated GAMS code.
- #978: Fix the domain of operations over an indexed subset.

CI/CD changes
-------------
- #965: Switch the build backend from setuptools to uv_build.

Miscellaneous internal changes
------------------------------
- #985: Add projection_tolerance option.


