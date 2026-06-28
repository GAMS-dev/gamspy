GAMSPy 1.25.0 (2026-06-27)
==========================

New features
------------
- #485: ``Container.read`` now accepts a ``dict`` for ``symbol_names`` to rename symbols on read. Keys are the symbol names in the source (GDX file or container) and values are the names of the symbols to create in the container, e.g. ``m.read(path, symbol_names={"X": "A"})``.
- #813: Added the new ``gamspy.formulations.sddp`` module implementing Stochastic Dual Dynamic Programming for multi-stage stochastic LPs.

Improvements in existing functionality
--------------------------------------
- #830: Speed up frozen solves. Modifiable records are now written directly into the in-memory GAMS database (GMD), removing the intermediate GAMS Transfer container. This cuts frozen solve time by roughly 20-35% on large models.
- #849: Keep the order of expressions when it involves conditions on the left.


