GAMSPy 1.28.1 (2026-10-02)
==========================

New features
------------
- #911: Added ``indicator`` formulation to ``gamspy.formulations`` for modeling indicator constraints, either reformulated or passed to the solver natively.
- #913: ``gp.If``, ``gp.ElseIf`` and ``gp.Else`` can now be used on the outermost level, and ``gp.If`` and ``gp.ElseIf`` accept an optional ``container`` argument.

Improvements in existing functionality
--------------------------------------
- #898: ``gamspy list solvers`` now shows a reduced, more readable table of solvers and tools, with a status column (licensed/installed) and one column per supported problem type, instead of a plain solver-to-problem-type listing.

Bug fixes
---------
- #904: Indexing a symbol with a set that an enclosing operation binds to a literal element, e.g. ``Sum(i["label"], p[i])``, no longer raises a false domain violation.
- #910: Corrected the equation list for the `cperfect` model and several incorrect `import` statements in the inlined example models.
- #914: Model attributes are now set after solving a model inside ``gp.Loop``, ``gp.For`` or ``gp.While``.

Improved documentation
----------------------
- #901: Rewrite the section on logical operators for better readability.

Deprecations
------------
- #898: The ``examiner2`` solver is deprecated and will be removed in a future release. ``Model.solve(solver="examiner2")`` now emits a ``DeprecationWarning`` and ``gamspy install solver examiner2`` prints a deprecation warning; use ``examiner`` instead.
  ``gamspy list solvers --installables`` (``-i``) is deprecated and will be removed in a future release; it now prints a deprecation warning. Use ``gamspy list solvers --all`` instead.

Miscellaneous internal changes
------------------------------
- #909: Move the socket connection to the GAMS process behind an execution engine abstraction to prepare the way for adding alternative local execution engines in the future.


