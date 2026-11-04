GAMSPy 1.29.3 (2026-11-04)
==========================

New features
------------
- #863: Expose the commands of the GAMSPy CLI as Python functions in ``gamspy.cli``, e.g. ``gamspy.cli.install_solver("highs")``.
- #937: Support listing, cancelling and tagging GAMS Engine jobs, reading text and stream entries, and querying the version, usage, quotas and namespace models.
- #940: Implement ``logical_and``, ``logical_or``, ``logical_not``, ``logical_xor``, ``logical_imp`` and ``logical_eqv`` for modeling logical relations between binary variables.
- #950: Add an optional ``report`` argument to ``GUSSScenarioDict.add_options`` that collects per-scenario model attributes, such as ``ModelStat`` and ``SolveStat``, into a parameter after each GUSS solve.

Bug fixes
---------
- #950: Prevent SDDP from using unsuccessful solve results when building cuts, measuring bounds, simulating, or querying a policy.
- #954: Respect the solver set for the problem type (e.g. ``lp``) in container options or solve options when ``solver`` is not given.

CI/CD changes
-------------
- #951: Install licenses and solvers in tests into a temporary license file and a fresh virtual environment instead of the active environment.
- #953: Publish to PyPI from GitHub Actions via trusted publishing so releases carry provenance.


