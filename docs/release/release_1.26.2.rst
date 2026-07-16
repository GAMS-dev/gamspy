GAMSPy 1.26.2 (2026-07-16)
==========================

New features
------------
- #711: Implements `toGraph()`` method to expressions, operations, and symbols. ``toGraph`` returns a ``graphviz.Digraph`` 
  of the underlying expression tree for inspection. For an ``Equation`` this graphs the definition of the equation and 
  for other symbols its definition. ``graphviz`` is added as an optional dependency.

Bug fixes
---------
- #741: Fixes a bug where creating a symbol with empty records from an empty iterable (e.g. ``[]``, ``set()``, ``range(0)``) raised a ``ValueError`` about inconsistent dimensionality. 
  Empty iterables are now accepted and produce a symbol with no records.
- #862: Fixes a bug where indexing a symbol with a set that is a superset of the declared domain raised a ``ValidationError``, even though GAMS accepts it. Domain compatibility is now checked in both directions, so both subset and superset index sets on the same domain chain are allowed (e.g. indexing ``GgRtpc(r, sub_t)`` with ``Rtpc(r, t)`` where ``sub_t`` is a subset of ``t``).

Improved documentation
----------------------
- #857: Adds a "How SDDP Works" page to the SDDP documentation: the cut algebra, the forward and backward passes, and the lower bound.


