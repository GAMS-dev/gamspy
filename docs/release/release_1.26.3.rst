GAMSPy 1.26.3 (2026-08-03)
==========================

New features
------------
- #867: Add cut selection to the ``sddp`` module; keeps only the cuts from the most recent ``N`` iterations and deactivates the older ones, bounding the size of the subproblems.
- #871: Add `gamspy.sparse`` to mark the right-hand side of an assignment as a sparse assignment. 
  ``p[i] = gp.sparse(rhs)`` generates the GAMS sparse assignment ``p(i) $= rhs;`` which 
  only assigns where the right-hand side is non-zero. It is equivalent to ``p[i].where[rhs] = rhs`` 
  but evaluates the right-hand side only once, which can be significantly faster if the right-hand 
  side is expensive to evaluate.

Improvements in existing functionality
--------------------------------------
- #870: Move SDDP API from the top-level ``gamspy.formulations`` namespace to the dedicated ``gamspy.formulations.sddp`` namespace.

Bug fixes
---------
- #865: Fix missing parentheses in set.lag operations when the jump is a multi-term expression.
- #872: Fix a false negative domain validation when a lag/lead operation is nested in the index of another set.


