GAMSPy 1.15.1 (2025-08-30)
==========================

New features
------------
- #642: Add RandomForest formulation.
- #655: Add GradientBoosting formulation.
- #682: - Introduce a convenience function for `tanh` in gamspy.math.activations to generate the output variable and equations.
  - Add `init_weights` parameter in `make_variable` for `Linear`, `Conv1d` and `Conv2d` formulations.

Improvements in existing functionality
--------------------------------------
- #686: Allow specifying checkout duration through gamspy retrieve command to allow checking out network licenses.

Improved documentation
----------------------
- #642: Add docs for RandomForest formulation.
- #655: Add docs for GradientBoosting formulation.
- #682: Update documentation for training NNs to showcase formulations.
- #688: Update documentation for ML formulations.

