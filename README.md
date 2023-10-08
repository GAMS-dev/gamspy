<div align="center">
  <img src="https://www.gams.com/img/gams_logo.svg"><br>
</div>

-----------------

# gamspy: algebraic modeling interface to GAMS

## Installation

```sh
pip install gamspy
```

## What is it?

**GAMSPy** is a mathematical optimization package that combines the power of the high performance GAMS execution system
and flexibility of the Python language. It includes all GAMS symbols (Set, Alias, Parameter, Variable, and
Equation) to compose mathematical models, a math package, and various utility functions.


## Design Philosophy
GAMSPy makes extensive use of "vectorization" -- the absence of any explicit looping, indexing, etc., in native Python.
These things are taking place, of course, just “behind the scenes” in optimized, pre-compiled C code.

Vectorized code has many advantages:

  - Results in more concise Python code -- avoids inefficient and difficult to read for loops
  - Closely resembles standard mathematical notation
  - Easier to read
  - Fewer lines of code generally means fewer bugs


## Main Features
Here are just a few of the things that **GAMSPy** does well:

  - Specify model algebra in Python natively
  - Combines the flexibility of Python programming flow controls and the power of model specification in GAMS
  - Test a variety of solvers on a model by changing only one line

## Documentation
The official documentation is hosted on [GAMSPy Readthedocs](https://gamspy.readthedocs.io/en/latest/index.html).

## Getting Help

For usage questions, the best place to go to is [GAMS](https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html).
General questions and discussions can also take place on the [GAMS World Forum](https://forum.gamsworld.org).

## Discussion and Development
If you have a design request or concern, please write to support@gams.com.
