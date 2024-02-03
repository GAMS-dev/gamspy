GAMSPy 0.11.8
=============

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