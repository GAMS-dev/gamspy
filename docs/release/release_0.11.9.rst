GAMSPy 0.11.9
=============

- General
  
  - Fix relative path issue of GAMS Engine backend.
  - Use $loadDC instead of $load to better catch domain violations.
  - Bypass constructor while creating a Container copy.
  - Do not execute_unload in case there is no dirty symbols to unload.
  - Update the behavior of `gamspy install/uninstall license`.
  - Implement GAMS Engine Client and consolidate NeosClient and EngineClient into one argument in solve.
  - Fix finding variables to mark in power and sameAs operations.

- Testing
  
  - Add test for GAMS Engine extra model files with incorrect relative path.
  - Add tests for new GAMS Engine Client.
  - Add a test to catch domain violation.
  - Remove declaration of objective variables and functions and add the equations into Python variables.
  - Add a new test to verify the license installation/uninstallation behavior.
  - Add a test to find variables in power operation.

- Documentation
  
  - Add a note in model documentation to warn about relative path requirement of GAMS Engine.
  - Add documentation for solving models asynchronously with GAMS Engine.
  - Modify model library table generation script to add more information and better table styling.