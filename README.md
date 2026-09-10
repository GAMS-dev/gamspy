![plot](https://github.com/GAMS-dev/gamspy/blob/develop/docs/_static/gamspy_logo.png?raw=true)

-----------------
[![PyPI version](https://img.shields.io/pypi/v/gamspy.svg?maxAge=3600)](https://gamspy.readthedocs.io/en/latest/)
[![Downloads](https://static.pepy.tech/badge/gamspy)](https://pepy.tech/project/gamspy)
[![Supported versions](https://img.shields.io/pypi/pyversions/gamspy.svg)](https://pypi.python.org/pypi/gamspy)
[![Documentation Status](https://readthedocs.org/projects/gamspy/badge/?version=latest)](https://gamspy.readthedocs.io/en/latest/)

# GAMSPy: Mathematical optimization in Python

GAMSPy is an open-source Python package for building and solving mathematical optimization models. Its algebraic, indexed API supports linear, mixed-integer, nonlinear, quadratically constrained, and complementarity models while fitting naturally into Python data and analytics workflows.

## Installation

```sh
pip install gamspy
```

GAMSPy requires Python 3.10 or newer. Installation includes the local execution runtime and a set of default solvers, so you do not need to install GAMS separately to run the example below. The included demo license is sufficient for this small model.

## Your first model

This production-mix model chooses how many desks and chairs to make to maximize profit within 40 available labor hours:

```python
import gamspy as gp

m = gp.Container()

products = gp.Set(m, records=["desk", "chair"])
profit = gp.Parameter(
    m,
    domain=products,
    records=[("desk", 100), ("chair", 40)],
    description="profit per unit",
)
hours = gp.Parameter(
    m,
    domain=products,
    records=[("desk", 4), ("chair", 1)],
    description="labor hours per unit",
)
production = gp.Variable(
    m,
    domain=products,
    type="Positive",
    description="units to produce",
)

capacity = gp.Equation(m)
capacity[...] = gp.Sum(products, hours[products] * production[products]) <= 40

model = gp.Model(
    m,
    equations=[capacity],
    problem=gp.Problem.LP,
    sense=gp.Sense.MAX,
    objective=gp.Sum(products, profit[products] * production[products]),
)
model.solve()

print(f"Maximum profit: {model.objective_value:.0f}")
```

```text
Maximum profit: 1600
```

The same workflow—define indexed data, variables, equations, and an objective, then solve—scales to larger and more complex models.

## Why GAMSPy

- **Algebraic, indexed modeling:** Express whole families of variables and constraints over sets instead of constructing each scalar expression with Python loops. Model definitions stay close to their mathematical notation.
- **Multiple problem classes:** Build linear, mixed-integer, nonlinear, quadratically constrained, complementarity, and related model types with a consistent API.
- **Choice of solver:** Use compatible installed solvers without rewriting the model. GAMSPy provides access to both open-source and commercial solvers, subject to their availability and license terms.
- **Python workflows:** Load model data from Python lists, NumPy arrays, or pandas objects, and work with results as pandas DataFrames alongside the rest of the Python ecosystem.

## Licensing and free use

The modeling package and the components used to execute and solve a model have different licensing considerations:

- **GAMSPy source code:** The Python source in this repository is available under the [MIT License](https://github.com/GAMS-dev/gamspy/blob/develop/LICENSE).
- **Demo license:** A time- and size-limited demo license is included with GAMSPy and can solve small models such as the example above. See the [demo license limitations](https://www.gams.com/latest/docs/UG_License.html#UG_License_Additional_Solver_Limits) for current details.
- **Free personal license:** Individuals can generate a free license for non-commercial learning, experimentation, hobby projects, and prototypes. It includes a selection of free and open-source solvers; see the [free-license overview](https://www.gams.com/free-licenses/) for the applicable terms.
- **Free academic GAMSPy license:** Eligible students, teachers, researchers, and academic staff can obtain a free license for academic, non-commercial, non-production use. It includes selected commercial solvers without model-size limits for qualifying use. See the current [included-solvers overview](https://www.gams.com/free-licenses/solvers/).
- **Commercial license:** Commercial and production use requires an appropriate license. Contact [sales@gams.com](mailto:sales@gams.com) or use the [GAMS contact form](https://www.gams.com/contact/) and select **Licensing**.

Sign up and generate a free personal or academic license at the [GAMS Portal](https://portal.gams.com/), then follow the [GAMSPy license installation instructions](https://gamspy.readthedocs.io/en/latest/user/installation.html#licensing).

Solver availability depends on both the installed solver package and the active license. Runtime and solver components may have terms separate from the MIT-licensed Python source.

## Relationship to GAMS

GAMSPy builds on GAMS model-generation and execution technology. You write and manage the model in Python; GAMSPy translates its algebraic representation for the execution system and dispatches it to a compatible solver. No knowledge of the GAMS language is required, and the standard `pip install gamspy` installation provides the local runtime needed to get started.

GAMS and GAMSPy licenses are separate. An existing GAMS license is not automatically a GAMSPy license; see the [installation and licensing guide](https://gamspy.readthedocs.io/en/latest/user/installation.html) for the available options and exceptions.

## Learn more

- [Documentation](https://gamspy.readthedocs.io/en/latest/)
- [Quick start guide](https://gamspy.readthedocs.io/en/latest/user/notebooks/trnsport.html)
- [Open the quick start in Colab](https://colab.research.google.com/github/GAMS-dev/gamspy/blob/develop/docs/user/notebooks/trnsport_colab.ipynb)
- [Example model library](https://github.com/GAMS-dev/gamspy-examples)
- [API reference](https://gamspy.readthedocs.io/en/latest/reference/index.html)
- [Licensing information](https://www.gams.com/free-licenses/)

## Community and contributing

For usage questions, visit the [GAMSPy documentation](https://gamspy.readthedocs.io/en/latest/) or the [GAMS Forum](https://forum.gams.com/). Report bugs and request features through the [GitHub issue tracker](https://github.com/GAMS-dev/gamspy/issues). Contributions are welcome; see the [contributing guide](https://github.com/GAMS-dev/gamspy/blob/develop/CONTRIBUTING.md) to get started.

## See GAMSPy in action

https://github.com/GAMS-dev/gamspy/assets/25618191/af91659c-408d-4f4c-a226-dc79e142a62f
