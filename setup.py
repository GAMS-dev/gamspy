from setuptools import setup, Distribution
import platform
import os


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def get_minigams_path():
    user_os = platform.system().lower()
    minigams_path = "minigams" + os.sep + user_os

    if user_os == "darwin":
        minigams_path += f"_{platform.machine()}"

    minigams_path += os.sep + "*"
    return minigams_path


classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering",
]

long_description = """
<div align="center">
  <img src="https://www.gams.com/img/gams_logo.svg"><br>
</div>

-----------------

# gamspy: algebraic modeling interface to GAMS

## What is it?

**gamspy** is a mathematical optimization package that combines the power of the high performance GAMS execution system
and flexibility of the Python language. It includes all GAMS symbols (Set, Alias, Parameter, Variable, and
Equation) to compose mathematical models, a math package, and various utility functions.


## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Main Features](#main-features)
- [Dependencies](#dependencies)
- [License](#license)
- [Documentation](#documentation)
- [Getting Help](#getting-help)
- [Discussion and Development](#discussion-and-development)


## Design Philosophy
GAMSPy makes extensive use of "vectorization" -- the absence of any explicit looping, indexing, etc., in native Python.
These things are taking place, of course, just “behind the scenes” in optimized, pre-compiled C code.

Vectorized code has many advantages:

  - Results in more concise Python code -- avoids inefficient and difficult to read for loops
  - Closely resembles standard mathematical notation
  - Easier to read
  - Fewer lines of code generally means fewer bugs


## Main Features
Here are just a few of the things that **gamspy** does well:

  - Specify model algebra in Python natively
  - Combines the flexibility of Python programming flow controls and the power of model specification in GAMS
  - Test a variety of solvers on a model by changing only one line

## Dependencies
GAMSPy has dependencies to other GAMS Python packages, specifically:

  - [gamsapi](www.cnn.com)
  - [minigams](www.cnn.com)

As well as third party packages:

  - [Pandas](https://pandas.pydata.org/)
  - [NumPy](https://numpy.org/)


```sh
# from PyPI
pip install gamspy
```

Might need to add in other installation notes (and fix links to documentation).


## Documentation
The official documentation is hosted on gams.com.

## Getting Help

For usage questions, the best place to go to is [GAMS](https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html).
General questions and discussions can also take place on the [GAMS World Forum](https://forum.gamsworld.org).

## Discussion and Development
If you have a design request or concern, please write to support@gams.com.
"""


setup(
    name="gamspy",
    version="0.9.0",
    description="Algebraic modeling interface to GAMS",
    url="http://www.gams.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GAMS Development Corporation",
    author_email="support@gams.com",
    python_requires=">=3.8",
    project_urls={
        "Documentation": "https://www.gams.com/latest/docs/API_PY_OVERVIEW.html"
    },
    packages=["gamspy"],
    classifiers=classifiers,
    package_dir={"gamspy": "src/gamspy"},
    distclass=BinaryDistribution,
    install_requires=["pandas>=1.5,<=2.1"],
)
