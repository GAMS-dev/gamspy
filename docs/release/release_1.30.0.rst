GAMSPy 1.30.0 (2026-11-12)
==========================

New features
------------
- #855: Added non-blocking solves via the new ``Model.solveAsync`` method, which submits the 
  solve and immediately returns a ``SolveHandle``. Multiple instances of the same model 
  can be in flight in parallel.
- #876: Add ``DomainViolationError``, the ``drop_domain_violations`` argument and ``findDomainViolations()`` to handle records outside a symbol's domain.
- #942: Add ``Model.getModelInstance`` to inspect the generated model instance.
- #959: Add reify formulation to gamspy.formulations for creating a binary variable that indicates whether a constraint holds.
- #960: Add ``solver`` and ``solver_options`` arguments to ``SDDP`` and ``SDDP.load``
  to configure all SDDP model solves.

Improvements in existing functionality
--------------------------------------
- #961: Install and uninstall add-on solvers, and determine the installed solvers and the solvers a license covers through gamspy_base instead of the GAMSPy CLI.

Bug fixes
---------
- #956: Fix ``gamspy uninstall solver`` failing when the GAMSPy user directory does not exist.
- #962: Fix zero-state propagation between stages in SDDP training and simulation.

Deprecations
------------
- #855: Deprecated ``is_blocking=False`` on ``EngineClient`` and ``NeosClient``. Constructing a 
  client with it now emits a ``DeprecationWarning``; use ``Model.solveAsync(backend=..., client=...)`` 
  instead, which returns a ``SolveHandle`` that loads the solution back into the model.
- #876: Deprecate the ``DROP_DOMAIN_VIOLATIONS`` option in favor of the ``drop_domain_violations`` argument.

Dependencies
------------
- #961: Upgrade the minimum version of gamspy_base and gamsapi to 54.7.0.

CI/CD changes
-------------
- #952: Replace the unmaintained detect-secrets pre-commit hook with betterleaks and add a hook that flags code run at interpreter startup from site-packages.


