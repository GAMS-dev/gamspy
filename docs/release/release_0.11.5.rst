GAMSPy 0.11.5
=============

- General
  
  - Verify dimensionality of the symbol and the given indices to provide better error messages.
  - Allow Model object to also accept tuple of equations.
  - List available and installed solvers in alphabetically sorted order.
  - Fix adding autogenerated equations twice. 
  - Generate unique names for the autogenerated variables and equations.
  - Add __str__ and __repr__ to Model.
  - Allow literals in sameAs operation.
  - Make Number operable.
  - Add more data validation functions.
  - Clear autogenerated symbols from the container if there is an exception.
  - Fix Alias bug while preparing the modified symbols list.

- Testing
  
  - Add tests to check if incompatible dimensionality throws exception.
  - Test validation errors.
  - Allow providing system directory for the tests via environment variable.

- Documentation
  
  - Add documentation for `matches` argument of Model.
