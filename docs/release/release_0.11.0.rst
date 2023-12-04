GAMSPy 0.11.0
=============

- General

  - Implement GAMS MIRO integration.
  - Generate expression representation as soon as it is created to avoid tall recursions.
  - Find variables in equations by iteratively traversing instead of doing recursion.

- Documentation
  - Add documentation of GAMS MIRO integration.
  - Add documentation for NEOS backend.

- Testing
  
  - Add tests for GAMS MIRO.
  - Add NEOS Server as a backend to solve models.
  - Add tests for NEOS backend.
