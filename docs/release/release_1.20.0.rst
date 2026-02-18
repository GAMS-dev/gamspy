GAMSPy 1.20.0 (2026-02-18)
==========================

Improvements in existing functionality
--------------------------------------
- #765: Return FormulationResult instance for Neural Network formulations (pooling, convolution, and linear layers).
- #772: Allow specifying gdx and python files as an option in 'gamspy mps2gms' command. Add '--compress' option to 'gamspy mps2gms'.

Bug fixes
---------
- #777: Fix model attribute updates for frozen models.

Improved documentation
----------------------
- #764: Sync tabs in embed neural networks docs.
- #771: Add a FAQ entry about running model instances in MPS or LP format.

Dependencies
------------
- #774: Upgrade gamspy_base and gamsapi dependency ranges.

Miscellaneous internal changes
------------------------------
- #768: Use pytest's tmp_path for tests instead of creating temp directories manually.

