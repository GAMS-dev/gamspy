GAMSPy 0.14.5
-------------
- General

  - Retry login with exponential backoff in GAMS Engine backend.
  - Allow to set all model attributes that can be set before solve in GAMS.
  - Fix equation listing, variable listing parsing when listing file is specified.

- Testing

  - Use contextmanager to create atomic conda environments.
  - Add tests for model attribute options.

- Documentation

  - Fix links in the api reference.
  - Add an example that shows how to embed NN to an optimization problem.
