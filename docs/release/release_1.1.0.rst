GAMSPy 1.1.0
------------

Release Date: 30.10.2024

- General

  - Allow printing the records of variable/equation attributes with a print(variable.attribute[domain].records) syntax.
  - Allow printing the records of a subset of a parameter with print(parameter['literal'].records) syntax.
  - Allow printing the records of a subset of a set with print(set['literal'].records) syntax.
  - Update variable/equation attribute domains on addGamsCode call.
  - Show log file instead of listing file on solve statements with NEOS backend.
  - Add Linear layer formulation
  - Fix minor bug of domain conflict in batched matrix multiplication
  - Improve the error messages of the thrown exceptions in case the user provide a model option at Container creation time.
  - Do not allow models with the same name to override each other.
  - Upgrade gamspy_base and gamsapi versions to 48.2.0.

- Testing

  - Fix race conditions in the pipeline.
  - Remove redundant setRecords in gapmin.py example.
  - Add sq.py model to the test model suite.
  - Update hansmge model.
  - Fix lower bound in reshop model.
  - Add tests for the Linear layer
  - Add a script to measure the overhead of GAMSPy and Python in general for each model in the model library.

- Documentation
  
  - Add documentation for the Linear layer
  