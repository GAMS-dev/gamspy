GAMSPy 1.6.0
------------

Release Date: 23.02.2025

- General

  - Upgrade pre-commit dependencies.
  - Enhance bound propagation for `AvgPool2d` class.
  - Allow adding debug options to Options objects.
  - Allow starting from a GAMS restart file.
  - Allow registering metadata to symbols via `symbol._metadata` attribute.
  - Fix solver option format of HIGHS, SHOT, SOPLEX and SCIP.
  - Allow dumping gams state on `.toGams` call.
  - Allow indexing into symbols with integers.
  - Add `bypass_solver`, `cutoff`, and `default_point` options.
  - Replace conda, pip and virtualenv with uv in ci pipelines.
  - Provide ssl context explicitly for NEOS backend.
  - Add configurable options via set_options and get_option.
  - Fix bug in an edge case of the vector-matrix multiplication.

- Testing

  - Add an lp and a qcp benchmark for performance comparison.

- Documentation

  - Add CNNs to embedding Neural Network documentation.
