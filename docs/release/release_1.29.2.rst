GAMSPy 1.29.2 (2026-10-27)
==========================

New features
------------
- #565: Add ``unicodeRepr`` to symbols, expressions, and operations to represent them in mathematical notation with Unicode characters, and ``Model.toUnicode`` to represent a whole model like ``Model.toLatex`` does.
- #923: Add an optional ``restrict`` argument to piecewise linear formulations and ``pwlinear`` to apply PWL relationships only to selected indexed entries.

Improvements in existing functionality
--------------------------------------
- #720: ``Problem``, ``Sense``, ``EquationType``, ``VariableType``, and ``FileFormat`` are now ``StrEnum`` subclasses, so their members compare equal to their string values.
- #945: Conv1d, Conv2d, MaxPool2d, MinPool2d and AvgPool2d window set creation improvements.

Bug fixes
---------
- #938: Discard the statements of context managers such as ``gp.Loop``, ``gp.For`` etc. 
  that raised an exception instead of executing them later.
- #939: Stop changing the attributes of user given options object in place as it might affect the subsequent solves.
- #941: Stop leaving temporary ``autotemp`` symbols in the container when evaluating the records of an expression, operation, condition or implicit symbol fails.
- #944: Fix compilation errors for symbol and model descriptions that contain double quotes.
- #946: Fix attribute assignments that are ignoring literal indices and transposition.
- #947: Fix ``gp.math.trace`` modifying the domain of its input symbol.
- #948: Fix keeping the pre-freeze values of certain variables and equations after a frozen solve.


