![plot](https://github.com/GAMS-dev/gamspy/blob/develop/docs/_static/gamspy_logo.png?raw=true)

-----------------
[![PyPI version](https://img.shields.io/pypi/v/gamspy.svg?maxAge=3600)](https://gamspy.readthedocs.io/en/latest/)
[![Downloads](https://static.pepy.tech/badge/gamspy)](https://pepy.tech/project/gamspy)
[![Documentation Status](https://readthedocs.org/projects/gamspy/badge/?version=latest)](https://gamspy.readthedocs.io/en/latest/)

# GAMSPy: Algebraic Modeling Interface to GAMS

## Installation

```sh
pip install gamspy
```

## What is it?

**gamspy** is a mathematical optimization package that combines the power of the high performance GAMS execution system
and flexibility of the Python language. It includes all GAMS symbols (Set, Alias, Parameter, Variable, and
Equation) to compose mathematical models, a math package, and various utility functions.

## Documentation
The official documentation is hosted on [GAMSPy Readthedocs](https://gamspy.readthedocs.io/en/latest/index.html).

## Design Philosophy
GAMSPy makes extensive use of set based operations -- the absence of any explicit looping, indexing, etc., in native Python.
These things are taking place, of course, just “behind the scenes” in optimized, pre-compiled C code.

Set based approach has many advantages:

  - Results in more concise Python code -- avoids inefficient and difficult to read for loops
  - Closely resembles standard mathematical notation
  - Easier to read
  - Fewer lines of code generally means fewer bugs


## Main Features
Here are just a few of the things that **gamspy** does well:

  - Specify model algebra in Python natively
  - Combines the flexibility of Python programming flow controls and the power of model specification in GAMS
  - Test a variety of solvers on a model by changing only one line

## Getting Help

For usage questions, the best place to go to is [GAMSPy Documentation](https://gamspy.readthedocs.io/en/latest/index.html).
General questions and discussions can also take place on the [GAMSPy Discourse Platform](https://forum.gams.com/c/gamspy-help).
