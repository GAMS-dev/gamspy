GAMSPy 1.27.0 (2026-08-21)
==========================

New features
------------
- #676: Allow filtering assignments by comparing a set or alias to an element label with ``==`` and ``!=``. For example, ``a[i].where[i == "i1"] = 0`` is now 
  equivalent to ``a[i].where[i.sameAs("i1")] = 0`` (and ``i != "i1"`` to ``~i.sameAs("i1")``). Expose the universe sentinel as ``gamspy.UNIVERSE``. Use 
  it wherever a domain is expected to leave that position undomain-checked, e.g. ``gp.Parameter(m, "misc", domain=[gp.UNIVERSE, r])``. It is now the 
  recommended way of specifying the GAMS universal set in symbol constructors.
- #760: Enhance piecewise linear formulations with heterogeneous indexed data,
  broadcasting across omitted domains, rays, and overlapping-chain support in
  Interval and DLog, and add a container-independent curve API through
  ``PWLCurve`` and ``pwlinear`` for selecting Interval, Convexity, or DLog.
- #877: Add NVIDIA cuOpt solver as an addon solver.

Improvements in existing functionality
--------------------------------------
- #598: Improve the performance of the frozen solves with a new hibernate mode. Generate records without materializing the dense cartesian product of the domains to avoid out of memory issues.

Bug fixes
---------
- #873: Quote the values of SCIP string options (e.g. ``solver_options={"lp/solver": "highs"}``) since SCIP's option file parser requires them to be enclosed in double quotes. 
  Char, and bool values as well as already quoted values are left untouched.

Deprecations
------------
- #676: Using the bare string ``"*"`` as a domain is soft-deprecated in favor of the new ``gamspy.UNIVERSE`` sentinel.
  ``"*"`` is still accepted everywhere and behaves identically (no warnings yet) but it may start emitting a
  ``DeprecationWarning`` in a future release, so prefer ``gamspy.UNIVERSE`` over ``"*"``.


