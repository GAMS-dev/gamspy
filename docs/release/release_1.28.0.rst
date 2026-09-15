GAMSPy 1.28.0 (2026-09-15)
==========================

New features
------------
- #880: Add the package wide option ``STRICT_POWER_OPERATOR`` which makes ``**`` generate ``rPower`` for every exponent, matching the GAMS ``**`` operator exactly.
- #889: Add typed symbol accessors to ``Container``: ``getSet``, ``getAlias``, ``getUniverseAlias``, ``getParameter``, ``getVariable``, ``getEquation``, and ``getSymbol(name, type)`` which narrows its return type to the requested symbol type.

Improvements in existing functionality
--------------------------------------
- #878: Internal performance improvements for `generateRecords`.
- #879: Setting records, converting them to and from numpy, and looking up UELs are all faster now, especially for symbols with large domains.
- #890: Every GAMSPy exception now derives from the new ``GamspyBaseException``, so a single ``except`` clause can handle any GAMSPy failure. ``GamspyException`` keeps covering only GAMS execution errors.
- #897: Added ``gamspy install solver --licensed`` to install every add-on solver permitted by the active license, and ``gamspy uninstall solver --unlicensed`` to uninstall installed add-on solvers no longer permitted by it.
- #899: Add the package wide option ``LICENSE_PATH`` (also settable via the ``GAMSPY_LICENSE_PATH`` environment variable) to use a license file without installing it.

Bug fixes
---------
- #880: Map ``**`` with a fractional exponent (e.g. ``x ** 2.9998``) to ``rPower`` instead of ``power``, which used to round it to the nearest integer.
- #881: Make SDDP training, simulation, and cut-selection bound evaluation work with the bundled demo license.
- #883: Merge the solver configuration of the ``cuopt`` release archive into the ``gamsconfig.yaml`` of the GAMS system directory instead of overwriting it.
- #887: An explicitly passed empty ``Container`` is no longer ignored inside a ``Container`` context manager and ``Alias``/``UniverseAlias`` redeclarations no longer emit a duplicate declaration.
- #890: whereMaxAbs now reports the record with the largest magnitude instead of None when that record holds a negative value.
- #893: ``set_options`` now raises ``ValidationError`` for an unrecognized option name instead of silently accepting it.
- #900: ``Model.freeze`` now raises a ``ValidationError`` if a variable of the model ends with the reserved ``_var`` suffix instead of letting GAMS fix that variable to zero, and a failed ``freeze`` no longer leaves the model in a frozen state.

Improved documentation
----------------------
- #890: Add examples in the docstrings of ``find*``, ``count*``, ``where*``, and ``shape`` functions.
- #902: Fix swapped left and right gradients in the piecewise linear functions example.

CI/CD changes
-------------
- #882: Remove deprecated prek auto-update command. Use prek update instead.

Miscellaneous internal changes
------------------------------
- #886: Refactor test suite to decrease duplication.
- #892: Type check the ``formulations`` modules with ``ty`` instead of excluding them.


