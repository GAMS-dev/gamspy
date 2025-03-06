GAMSPy 1.7.0
------------

Release Date: 06.03.2025

- General

  - Allow container serialization/deserialization.
  - Support an alternative syntax for operations. For example, x.sum() is equivalent to Sum(x.domain, x[x.domain]).
  - Fix a bug when starting from a GAMS restart file.
  - Allow propagating bounds of the output in `Conv2D` class.
  - Introduce `name_prefix` option to NN formulations for ease of debugging.

- Documentation

  - Add a section in FAQ about the compatibiltiy issues of the Python interpreter from the Microsoft Store.
  - Fix minor issue in embedding Neural Network documentation.

- Testing

  - Enforce the order of tests. Run unit tests first, and model library tests last.
  - Use spawn method for multiprocessing to avoid possible deadlocks with fork method.
