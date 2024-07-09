GAMSPy 0.13.2
=============

- General

  - Set the records of objective value in model instance solves. 
  - Allow using an environment variable to set the GAMS system directory (given environment variable will override the system directory even if the user provides a system directory argument to Container).
  - Use gdxSymbols commandline option instead of manually marking symbols dirty.
  - Add memory_tick_interval, monitor_process_tree_memory, and profile_file options.
  - Change the way to generate a GAMS model from a GAMSPy model.
  - Remove import_symbols argument for addGamsCode since it is not needed anymore.

- Documentation

  - Redirect model library page to gamspy-examples Github repo.
  - Update toGams docs.
  - Update doctest of addGamsCode.

- Testing

  - Add model instance tests that check the objective value.
  - Update system directory test to adjust to the environment variable support.
  - Add tests for profiling options.
  