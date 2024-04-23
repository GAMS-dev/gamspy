GAMSPy 0.12.3
=============

- General

  - Set log and listing file option relative to os.cwd instead of workspace.working_directory.
  - Simplify expression generation and fix incorrect expression data. 
  - Add logoption=4.
  - Add show_raw option to the generateGamsString function.

- Testing
  
  - Test relative path for listing file and log file creation options.
  - Update log option tests.
  - Add new tests for generateGamString.

- Documentation
  
  - Remove the remnants of .definition and .assignment syntax from documentation.
  - Fix the example in gamspy for gams users.
  - Add notes about the equivalent operation in GAMS to .where syntax in GAMSPy.
  - Update the documentation for debugging with generateGamsString.