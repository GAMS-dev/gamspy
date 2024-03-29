GAMSPy CHANGELOG
================

GAMSPy 0.12.2
-------------
- General
  - Add infeasibility_tolerance as a model attribute.
  - Make urllib3 a true dependency instead of an optional one.
  - Do not suppress compiler listing by default.
  - Improve the performance of model attribute loading.
  - Fix license path for model instance.
- Documentation
  - Add documentation about solver specific infeasibility options.

GAMSPy 0.12.1
-------------
- General
  - Fix dataframe column names of GAMS MIRO input tables.
  - Catch solve status errors and throw necessary exceptions.
  - __pow__ returns sqrt instead of rPower if the exponent is 0.5.
  - Deprecate delayed_execution mode.
  - Replace pylint, flake8 and black with ruff.
  - Implement /api/auth -> post, /api/auth/login -> post and /api/auth/logout -> post for GAMS Engine.
  - Allow dumping log file to arbitrary path.
  - Allow dumping listing file to arbitrary path.
  - Allow dumping gdx file to arbitrary path.
  - Disallow equation definitions without any equality sign.
  - Add calculate_infeasibilities function for variables, equations and models.
  - Add 'gamspy show license', and 'gamspy show base' commands.
  - Replace __del__ with atexit.register function.
- Testing
  - Replace cta PandasExcelReader and PandasExcelWriter with new ExcelReader and ExcelWriter from GAMS Connect correspondingly. 
  - Add a new model (Nurses) to the model library and the Notebook examples.
  - Add an AC optimal power flow (ACOPF) model to the model library.
  - Add a test to verify the generated string for power where the exponent is 0.5.
  - Add tests for /api/auth.
  - Add a test for creating log file with arbitrary name.
  - Add a test for creating lst file with arbitrary name.
  - Add a test for creating gdx file with arbitrary name.
  - Add tests for infeasibility calculations.
- Documentation
  - Remove FAQ about Google Colab (it is resolved) and add FAQ about Windows Defender.
  - Remove documentation for delayed execution mode.
  - Add an example for providing solver options.
  - Document CLI for gamspy show commands.

GAMSPy 0.12.0
-------------
- General
  - Implement GAMS MIRO integration.
  - Update minimum gamsapi and gamspy_base version to 46.1.0.
- Testing
  - Add tests for GAMS MIRO.
- Documentation
  - Add documentation of GAMS MIRO integration.
  
GAMSPy 0.11.10
-------------------------------------------------------------------------------
- General
  - Adapt debugging level to GAMS 46 debugging levels.
  - Adapt getInstalledSolvers to renaming of SCENSOLVER
- Testing
  - Add test for GAMS Engine extra model files with incorrect relative path.
  - Update the results of model instance tests (CONOPT3 -> CONOPT4).

GAMSPy 0.11.9
-------------------------------------------------------------------------------
- General
  - Fix relative path issue of GAMS Engine backend.
  - Use $loadDC instead of $load to better catch domain violations.
  - Bypass constructor while creating a Container copy.
  - Do not execute_unload in case there is no dirty symbols to unload.
  - Update the behavior of `gamspy install/uninstall license`.
  - Implement GAMS Engine Client and consolidate NeosClient and EngineClient into one argument in solve.
  - Fix finding variables to mark in power and sameAs operations.
- Testing
  - Add test for GAMS Engine extra model files with incorrect relative path.
  - Add tests for new GAMS Engine Client.
  - Add a test to catch domain violation.
  - Remove declaration of objective variables and functions and add the equations into Python variables.
  - Add a new test to verify the license installation/uninstallation behavior.
  - Add a test to find variables in power operation.
- Documentation
  - Add a note in model documentation to warn about relative path requirement of GAMS Engine.
  - Add documentation for solving models asynchronously with GAMS Engine.
  - Modify model library table generation script to add more information and better table styling.

-------------------------------------------------------------------------------
GAMSPy 0.11.8
-------------------------------------------------------------------------------
- General
  - Allow assigning VariableType enum or EquationType enum as an attribute after object creation for Equation and Variable.
  - Fix running twice on symbol declaration with records
  - Return better error messages for incorrectly provided solver, options, and output arguments.
  - Fix missing uels_on_axes argument in setRecords.
  - Start using pylint to improve code quality.
- Testing
  - Add tests for assigning type to Variable and Equation after creation.
  - Add models information at the top of each model's docstring.
  - Add tests for setRecords with uels on axes.
- Documentation
  - Add docs for translating GAMS Macros to GAMSPy.

-------------------------------------------------------------------------------
GAMSPy 0.11.7
-------------------------------------------------------------------------------
- General
  - Implement GAMS MIRO integration.
  - Allow variable/equation attribute assignment without any index.
  - Run GAMS on symbol declaration and setRecords.
  - Add debugging_level argument to Container.
  - Performance improvements (~25-30%).
- Testing
  - Add tests for GAMS MIRO.
  - Test scalar variable/equation assignment without any index.
  - Test uel order.
- Documentation
  - Add documentation of GAMS MIRO integration.
  - Document assigning to scalar variable/equation.
  - Update documentation of frozen solve (model instance in GAMS). 
  - Add documentation for debugging levels.

-------------------------------------------------------------------------------
GAMSPy 0.11.6
-------------------------------------------------------------------------------
- General
  - Support slices for indexing.
  - Fix unary operator for expressions
  - Fixes SpecialValues bug in expressions.
  - Fix the bug for nonbinding equations.
  - Fix discovery of variables in math operations.
  - Fix literal while checking for domain validation.
- Testing
  - Add tests for slicing and ellipsis.
  - Add tests for unary operator applied to expressions.
  - Add tests to verify the correctness of SpecialValues in expressions.
  - Add more tests for nonbinding equations.
- Documentation
  - Document indexing with slices and ellipsis.

-------------------------------------------------------------------------------
GAMSPy 0.11.5
-------------------------------------------------------------------------------
- General
  - Verify dimensionality of the symbol and the given indices to provide better error messages.
  - Allow Model object to also accept tuple of equations.
  - List available and installed solvers in alphabetically sorted order.
  - Fix adding autogenerated equations twice. 
  - Generate unique names for the autogenerated variables and equations.
  - Add __str__ and __repr__ to Model.
  - Allow literals in sameAs operation.
  - Make Number operable.
  - Add more data validation functions.
  - Clear autogenerated symbols from the container if there is an exception.
  - Fix Alias bug while preparing modified symbols list.
- Testing
  - Add tests to check if incompatible dimensionality throws exception.
  - Test validation errors.
  - Allow providing system directory for the tests via environment variable.
- Documentation
  - Add documentation for `matches` argument of Model.

-------------------------------------------------------------------------------
GAMSPy 0.11.4
-------------------------------------------------------------------------------
- General
  - Expose GTP special values via gp.SpecialValues
  - Fix NEOS bug when the previous workfile comes from another backend.
  - Optimize read function of Container by assigning the symbols' attributes directly while casting.
  - Remove autogenerated variable and equation from Container after each solve.
  - Recover dirty and modified states if the write is invoked by the user.
  - Do not expose cast_to_gamspy to user.
  - Abstract backends to allow easier extension.
  - Add compress, mode, eps_to_zero arguments to write
  - Add load_records, mode, and encoding arguments to read
- Documentation
  - Fix Variable attribute assignments in user guide.
  - Add more examples in docstrings.
  - Add docs for collecting the results of non-blocking NEOS Server solves.
- Testing
  - Test the special value usage in assignments for Parameter, ImplicitParameter and Operation (Sum, Smax, Smin, Product).
  - Add hansmpsge model to the model library.
  - Add tests for the new arguments of write
  - Add tests for the new arguments of read

-------------------------------------------------------------------------------
GAMSPy 0.11.3
-------------------------------------------------------------------------------
- General
  - Fix setRecords bug
  - Run after an equation is defined
- Testing
  - Fix incorrect order of setRecords in gapmin model
  - Fix domain violation in the unit tests revealed by the execution of 
  equation definitions in immediate mode.
  - Use gams_math.sqr instead of custom sqr function in tests.

-------------------------------------------------------------------------------
GAMSPy 0.11.2
-------------------------------------------------------------------------------
- General
  - Fix the bug in writing only modified symbols.
  - Return summary dataframe for all synchronous backends.
  - Fix the bug in using set, alias attributes in conditions.
- Documentation
  - Re-run notebooks to reflect the changes in solve summary.
- Testing
  - Add tests for the returned summary dataframe from solve.
  - Add tests for solve with trace options.

-------------------------------------------------------------------------------
GAMSPy 0.11.1
-------------------------------------------------------------------------------
- General
  - Fix missing atttributes of Alias such as .first, .last etc.
  - Fix global option bug
  - Display summary on Jupyter Notebook.
- Testing
  - Add tests for Alias attributes.

-------------------------------------------------------------------------------
GAMSPy 0.11.0
-------------------------------------------------------------------------------
- General
  - Generate expression representation as soon as it is created to avoid tall recursions.
  - Find variables in equations by iteratively traversing instead of doing recursion.
  - Add NEOS Server as a backend to solve models.
  - Fix domain for the equations that were specified in the constructor of the equation.
  - Check if the container of domain symbols of a symbol match with the symbol's container.
  - Check if the container is valid before running the model.
- Documentation
  - Add documentation for NEOS backend.
- Testing
  - Add NEOS Server as a backend to solve models.
  - Add tests for NEOS backend.
  - Add tests for equations that were defined in the constructor.
  - Add tests for checking the containers of domain symbols.

-------------------------------------------------------------------------------
GAMSPy 0.10.5
-------------------------------------------------------------------------------
- General
  - Fix the issue of not setting options that are set to 0 (bug fix)
- Testing
  - Remove duplicated equations in models for MCP models.

-------------------------------------------------------------------------------
GAMSPy 0.10.4
-------------------------------------------------------------------------------
- General
  - Fix not equals overload of Ord and Card operations (bug fix)
  - Refactor generation of GAMS string
- Documentation
  - Move doc dependencies to pyproject.toml

-------------------------------------------------------------------------------
GAMSPy 0.10.3
-------------------------------------------------------------------------------
- General
  - Allow creating log file in working directory.
  - Forbid extra arguments for pydantic models (Options, EngineCofig)
- Documentation
  - Update model options table
  - Update jupyter notebook examples
- Testing
  - Adapt tests to new Options class instead of using dictionary.

-------------------------------------------------------------------------------
GAMSPy 0.10.2
-------------------------------------------------------------------------------
- General
  - Write and read only dirty symbols instead of all symbols to improve performance (~30% improvement on running all model library models).
  - Make gdx file names thread safe by using uuid.
- Documentation
  - Fix api reference for inherited members.
  - Make execution modes and debugging section of container documentation a separate page.
- Testing
  - Add a new test for sending extra files to GAMS Engine.
  - Add scripts/atomic_conda_env.py to avoid race condition for parallel builds in the pipeline.

-------------------------------------------------------------------------------
GAMSPy 0.10.1
-------------------------------------------------------------------------------
- General
  - Fix ellipsis syntax bug for variable and equation attributes
  - Introduce Pydantic as a dependency for options and engine config validation
- Documentation
  - Change reference API structure so that each class has its own page
- Testing
  - Simplify reinstall.py script
  - Add tests for options
  - Update tests for symbol creation

-------------------------------------------------------------------------------
GAMSPy 0.10.0
-------------------------------------------------------------------------------

- Initial release.
