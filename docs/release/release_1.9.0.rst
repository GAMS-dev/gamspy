GAMSPy 1.9.0
------------

Release Date: 09.04.2025

- General

  - Validate solver options for most of the solvers. It can be disable through VALIDATE_SOLVER_OPTIONS option.
  - get the value of objective estimation and the number of nodes used after frozen solves
  - Add description argument for model objects.
  - Make GAMSPy symbols non-iterable.
  - Inherit output argument from the container in solve function if the output argument is not specified.
  - Start the renaming process (deprecation) of model_instance_options to freeze_options. GAMSPy 1.9.0 will throw a warning. GAMSPy 1.10.0 will throw an exception, and GAMSPy 1.11.0 will remove model_instance_options altogether.
  - Fix sense=feasibility bug of frozen models. 
  - Rename ModelInstanceOptions to FreezeOptions and add a warning for the usage of ModelInstanceOptions.
  - Add model.convert function to allow converting a GAMSPy model instance to different file formats.
  - Fix sense=feasibility bug of frozen models.
  - Fix static code analysis errors.
  - Do not validate equation definitions in case the container was restarted from a save file (.g00 file).
  - Propagate the output option of the container to `model.freeze`.
  - Raise warning in case the frozen solve is interrupted.
  - Improve the performance of symbol declarations without any records and declaration of 0 dimensional symbols with records.

- Documentation

  - Add additional instructions to deploy a GAMSPy/GAMS MIRO model.
  - Fix name mismatch between the argument name and the docstring of loadRecordsFromGdx function.

- Testing

  - Run all pre-commit hooks instead of running selectively.
