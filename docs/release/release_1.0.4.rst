GAMSPy 1.0.4
------------

Release Date: 17.10.2024

- General

  - Do not create a GDX file when it's not necessary. 
  - Do not carry solver options from the previous solve to the new solve.
  - Fix toGams bug of MathOp symbols.
  - Use symbol< syntax of GAMS to handle domain forwarding.
  - Add "same" and "valid" options for Conv2d padding.
  - Update dependencies. gamspy_base -> 48.1.1 and gamsapi -> 48.1.0.
  - Make minimum supported Python version 3.9 and add support for Python 3.13.

- Documentation

  - Fix documented type of model.solve_status.
  - Add num_equations attribute to the model page of user guide.
  - Add synchronization docs to reference api.

- Testing

  - Add one to one comparison tests with reference files in toGams tests.
  - Add tests for "same" and "valid" padding options of Conv2d.
  