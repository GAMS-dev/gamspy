GAMSPy 1.0.0rc1
---------------
- General

  - Fix starting from a loadpoint for GAMS Engine backend.
  - Fix solver options issue for GAMS Engine backend.
  - Fix solver options issue for NEOS backend.
  - Support external equation for GAMS Engine backend.
  - Change the behaviour of expert synch mode.
  - Update quick start guide with latex to pdf output.
  - Fix quote issue in paths.
  - Activation functions now return added equations as well.
  - skip_intrinsic option added for log_softmax.
  - Allow installing/uninstalling multiple solvers at once.
  - Make miro_protect an option.
  - Show a better help message on gamspy -h command.
  - Fix missing links in api reference.
  - Set default problem type as MIP instead of LP.
  - Allow UniverseAlias in assignments.

- Documentation

  - Add a warning about the manipulation of records via .records. 
  - Fix model attribute return type.
  