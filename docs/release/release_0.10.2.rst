GAMSPy 0.10.2
=============

- General

  - Write and read only dirty symbols instead of all symbols to improve performance.
  - Make gdx file names thread safe by using uuid.

- Documentation
  
  - Fix api reference for inherited members.
  - Make execution modes and debugging section of container documentation a separate page.

- Testing
  
  - Add a new test for sending extra files to GAMS Engine.
  - Add scripts/atomic_conda_env.py to avoid race condition for parallel builds in the pipeline.