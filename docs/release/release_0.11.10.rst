GAMSPy 0.11.10
==============

- General

  - Adapt debugging level to GAMS 46 debugging levels.
    - Users have to upgrade their gamspy_base version. Command to upgrade: `gamspy update` or `pip install gamspy_base --upgrade`.
  - Adapt getInstalledSolvers to renaming of SCENSOLVER

- Testing
  
  - Add test for GAMS Engine extra model files with incorrect relative path.
  - Update the results of model instance tests (CONOPT3 -> CONOPT4).
