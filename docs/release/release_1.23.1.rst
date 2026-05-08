GAMSPy 1.23.1 (2026-05-07)
==========================

New features
------------
- #794: Allow providing symbol_names argument as a dict in container.write().
- #803: Add Gaussian Error Linear Unit (GELU) activation function.

Improvements in existing functionality
--------------------------------------
- #794: Include UELs in the gdx file generated with container.write().
  Throw deprecation warning when 'mode' argument is provided in container.write for future deprecation.
- #807: Adjusted cta.py for correct usage of GAMSPy with GAMS Connect.

Bug fixes
---------
- #802: Fixed equality sign bug in certain multi condition expressions.
- #804: Fix domain validation for SetExpression objects as domain elements.

Miscellaneous internal changes
------------------------------
- #803: Add test for GELU activation function.
- #806: Add additional tests to make sure that GTP container to GP container conversion works as expected.


