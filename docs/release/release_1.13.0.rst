GAMSPy 1.13.0 (2025-07-15)
==========================

New features
------------
- #616: Implement container.writeSolverOptions to write solver option files to the working directory.
- #654: Implemented `TorchSequential` convenience formulation for embedding Sequential layers into GAMSPy.

Improvements in existing functionality
--------------------------------------
- #658: Add a gamspy option to disable solver validation. Useful for solvers that are added via gamsconfig.yaml.
- #659: Allow server and port specification for on-prem license servers.
- #660: Add `DROP_DOMAIN_VIOLATIONS` option.

Bug fixes
---------
- #657: Unbounded input in the RegressionTree caused the value of M to become `infinity`. To prevent this, limit M to 1e10.
- #663: Fix duplicate domain name issue in the MIRO contract.
  Fix symbol declaration without records for miro apps.
- #665: Allow PathLike objects for loadpoint option.
- #666: Fix set attributes records call.
- #667: Fix the bug in expert sync mode due to missing attribute.

Improved documentation
----------------------
- #654: Added docs for `TorchSequential` formulation.

