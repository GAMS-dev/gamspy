GAMSPy 0.11.0
=============

- General
  
  - Add NEOS Server as a backend to solve models.
  - Generate expression representation as soon as it is created to avoid tall recursions.
  - Find variables in equations by iteratively traversing instead of doing recursion.
  - Fix domain for the equations that were specified in the constructor of the equation.

- Documentation
  
  - Add documentation for NEOS backend.

- Testing
  
  - Add tests for NEOS backend.
  - Add tests for equations that were defined in the constructor.