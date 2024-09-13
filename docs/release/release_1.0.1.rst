GAMSPy 1.0.1
------------

Release Date: 13.09.2024

- General

  - Fix frozen solve with non-scalar symbols.
  - Fix the definition update problem while redefining an equation with definition argument.
  - Introduce default directories to keep license information on upgrade.
  - Add --existing-solvers and --install-all-solvers options for gamspy install solver.
  - Add --uninstall-all-solvers option for gamspy uninstall solver.
  - Show license path on gamspy show license command.
  - Simplify the implementation of the copy container operation.
  - Add Conv2d formulation for convenience
  - Map GAMSPy problem types to NEOS problem types before sending the job.
  - Upgrade gamspy_base and gamsapi versions to 47.6.0. 

- Testing

  - Add test for the frozen solve with non-scalar symbols.
  - Add a test to verify the behaviour of equation redefinition with definition argument.
  - Test the usage of a license that is in one of the default paths.
  - Fix the issue related to reading equation records from a gdx file.
  - Add tests to verify the records after reading them from a gdx file.
  - Add tests for installing/uninstalling solvers.
  - Add tests to verify correctness of Conv2d formulation
  - Add a test to verify GAMSPy -> NEOS mapping.
  - Add an execution error test.

- Documentation

  - Update the documentation of install/uninstall command line arguments.
  - Add a section for NN formulations