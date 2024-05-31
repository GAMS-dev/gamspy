GAMSPy 0.12.1
=============

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

- Testing

  - Replace PandasExcelReader and PandasExcelWriter with new ExcelReader and ExcelWriter from GAMS Connect correspondingly. 
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
