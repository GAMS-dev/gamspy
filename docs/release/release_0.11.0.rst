GAMSPy 0.11.0
=============

- General

  - Generate expression representation as soon as it is created to avoid tall recursions.
  - Find variables in equations by iteratively traversing instead of doing recursion.
  - Fix domain for the equations that were specified in the constructor of the equation.
  - Check if the container of domain symbols of a symbol match with the symbol's container.
  - Check if the container is valid before running the model.

- Documentation
  - Add documentation for NEOS backend.

- Testing
  
  - Add NEOS Server as a backend to solve models.
  - Add tests for NEOS backend.
  - Add tests for equations that were defined in the constructor.
  - Add tests for checking the containers of domain symbols.
  