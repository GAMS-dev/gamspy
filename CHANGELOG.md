GAMSPy CHANGELOG
================

-------------------------------------------------------------------------------
GAMSPy 0.10.2
-------------------------------------------------------------------------------
- General
  - Write and read only dirty symbols instead of all symbols to improve performance.
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