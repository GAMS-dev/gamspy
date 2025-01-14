GAMSPy 1.5.0
------------

Release Date: 10.01.2025

- General

  - Fix implicit parameter validation bug.
  - Allow the usage of Container as a context manager.
  - Allow propagating bounds to the output variable in `flatten_dims` method.
  - Add piecewise linear function formulations.
  - Migrate GAMSPy CLI to Typer.
  - Threads can now create a container since we register the signal only to the main thread.
  - Fix solver options bug in frozen solve.
  - Synchronize after read.
  - Upgrade gamspy_base and gamsapi dependencies.

- Testing

  - Lower the number of dices in the interrupt test and put a time limit to the solve.
  - Add tests for piecewise linear functions.

- Documentation

  - Install dependencies in the first cell of the example transportation notebook.
  - Add Formulations page to list piecewise linear functions and nn formulations.
  