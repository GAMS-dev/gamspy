GAMSPy 0.14.4
-------------
- General

  - Add container.in_miro flag to selectively load data.
  - Parse error message after verifying the return code for engine backend.
  - Fix the behaviour of Model if it's declared twice with objective function.
  - Update the error message of license error.
  - Fix output stream validation.
  - Fix exception on solve in case listing file is specified.
  - Add external equations support.
  - Do not raise exception in case GAMS Engine returns 308 on get_logs call.

- Testing

  - Add test for container.in_miro flag.
  - Add tests to simulate Jupyter Notebook behaviour.
  - Remove system_directory for tests.
  - Add a test which specifies the listing file and fails because the license does not allow to run the model.
  - Add tests for external equations support.
  - Add traffic model to the model library.

- Documentation

  - Document in_miro flag.
  - Add docstring for setBaseEqual.
  - Add section "External Equations" under Advanced documentation.
  - Add section "Extrinsic Functions" under Advanced documentation.
