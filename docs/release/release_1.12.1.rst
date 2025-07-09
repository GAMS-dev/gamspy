GAMSPy 1.12.1 (2025-07-07)
==========================

Improvements in existing functionality
======================================

- #658: Add a gamspy option to disable solver validation. Useful for solvers that are added via gamsconfig.yaml.

- #659: Allow server and port specification for on-prem license servers.

- #660: Add `DROP_DOMAIN_VIOLATIONS` option.


Bug fixes
=========

- #657: Unbounded input in the RegressionTree caused the value of M to become `infinity`. To prevent this, limit M to 1e10.

- #663: Fix duplicate domain name issue in the MIRO contract.
  Fix symbol declaration without records for miro apps.


