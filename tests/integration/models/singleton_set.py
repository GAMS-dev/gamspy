"""
## LICENSETYPE: Demo
## MODELTYPE: LP
## KEYWORDS: singleton set, matrix multiplication


Singleton Set Matrix Mult
-------------------------

Description:
    Starting with GAMS 51, singleton sets can be used in symbol domains
    within definitions. Multiplying a row vector by a column vector produces a
    1x1 matrix. Previously, assigning this result directly to a scalar without
    specifying an explicit domain would trigger a domain not controlled error.

    With the introduction of GAMS 51, you can now assign 1x1 matrices—or even
    higher-dimensional tensors—to scalars, provided the domains are defined
    using the `Dim` function.

Usage: python singleton_set.py
"""

from __future__ import annotations

import math

import numpy as np

from gamspy import Container, Equation, Model, Parameter, Variable
from gamspy.math import dim


def main():
    m = Container()
    scalar = Variable(m, "scalar_val")

    scalar_2 = Parameter(m, "scalar_val_2", records=-1.6)
    column_vector = Variable(m, "column_vector", domain=dim([2, 1]))
    column_vector.fx[...] = 0.4

    row_vector = Parameter(
        m,
        "row_vector",
        domain=dim([1, 2]),
        records=np.array([[0.5, 2]]),
    )

    assign_scalar = Equation(m, "assign_scalar")
    assign_scalar[...] = scalar == row_vector @ column_vector + scalar_2

    m = Model(
        m,
        name="test_problem",
        equations=[assign_scalar],
    )

    m.solve()
    output = float(scalar.toDense())
    assert math.isclose(output, -0.6), output
    print("Scalar value: ", scalar.toDense())


if __name__ == "__main__":
    main()
