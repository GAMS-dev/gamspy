GAMSPy CHANGELOG
================

-------------------------------------------------------------------------------
GAMSPy 0.11.0
-------------------------------------------------------------------------------
- General
  - Implement GAMS MIRO integration.
  - Generate expression representation as soon as it is created to avoid tall recursions.
  - Find variables in equations by iteratively traversing instead of doing recursion.
- Documentation
  - Add documentation of GAMS MIRO integration.
- Testing
  - Add tests for GAMS MIRO.

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
