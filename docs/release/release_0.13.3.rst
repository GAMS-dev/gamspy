GAMSPy 0.13.3
=============

- General
  
  - Change default solvers to 'CONOPT', 'CONVERT', 'CPLEX', 'GUSS', 'IPOPT', 'IPOPTH', 'KESTREL', 'NLPEC', 'PATH', and 'SHOT'
  - Fix the version of gamspy_base when "gamspy update" command is being executed.
  - Fix the order issue for Alias in toGams function.
  - Add exponential backoff for GAMS Engine logout api.
  - Add symbol validation for Ord operation.

- Testing

  - Update model library tests according to the new default solvers.
  - Add a test to verify that modifiable symbols cannot be in conditions for model instance runs.
  - Add new tests for symbol validation.
