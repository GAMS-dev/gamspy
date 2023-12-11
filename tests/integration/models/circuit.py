"""
Optimal design of an electrical circuit.

Ratschek, H., Rokne, J., A circuit design problem. J. Global Opt., 3,
1993, pp.501.

Neculai Andrei, "Models, Test Problems and Applications for
Mathematical Programming". Technical Press, Bucharest, 2003.
Application A34, pp.397.
"""
from __future__ import annotations

import os

import numpy as np

import gamspy.math as gams_math
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Variable


def main():
    cont = Container(
        delayed_execution=int(os.getenv("DELAYED_EXECUTION", False))
    )

    # SETS #
    n = Set(cont, name="n", records=[f"c{c}" for c in range(1, 5)])
    m = Set(cont, name="m", records=[f"r{r}" for r in range(1, 6)])

    # PARAMETER #
    g = Parameter(
        cont,
        name="g",
        domain=[m, n],
        records=np.array(
            [
                [0.4850, 0.7520, 0.8690, 0.9820],
                [0.3690, 1.2540, 0.7030, 1.4550],
                [5.2095, 10.0677, 22.9274, 20.2153],
                [23.3037, 101.7790, 111.4610, 191.2670],
                [28.5132, 111.8467, 134.3884, 211.4823],
            ]
        ),
    )

    # VARIABLES #
    x1 = Variable(cont, name="x1")
    x2 = Variable(cont, name="x2")
    x3 = Variable(cont, name="x3")
    x4 = Variable(cont, name="x4")
    x5 = Variable(cont, name="x5")
    x6 = Variable(cont, name="x6")
    x7 = Variable(cont, name="x7")
    x8 = Variable(cont, name="x8")
    x9 = Variable(cont, name="x9")
    x10 = Variable(cont, name="x10")
    obj = Variable(cont, name="obj")

    # EQUATIONS #
    e1 = Equation(cont, name="e1", type="regular", domain=[n])
    e2 = Equation(cont, name="e2", type="regular", domain=[n])
    e3 = Equation(cont, name="e3", type="regular", domain=[n])
    e4 = Equation(cont, name="e4", type="regular", domain=[n])
    e = Equation(cont, name="e", type="regular")
    eobj = Equation(cont, name="eobj", type="regular")

    e1[n] = (
        g["r4", n] * x2
        - x10
        + (1 - x1 * x2)
        * x3
        * (
            gams_math.exp(
                x5
                * (
                    g["r1", n]
                    - g["r3", n] * x7 / 1000
                    - g["r5", n] * x8 / 1000
                )
            )
            - 1
        )
        <= g["r5", n]
    )

    e2[n] = (
        -g["r4", n] * x2
        - x10
        - (1 - x1 * x2)
        * x3
        * (
            gams_math.exp(
                x5
                * (
                    g["r1", n]
                    - g["r3", n] * x7 / 1000
                    - g["r5", n] * x8 / 1000
                )
            )
            - 1
        )
        <= -g["r5", n]
    )

    e3[n] = (
        -g["r5", n] * x1
        - x10
        + (1 - x1 * x2)
        * x4
        * (
            gams_math.exp(
                x6
                * (
                    g["r1", n]
                    - g["r2", n]
                    - g["r3", n] * x7 / 1000
                    + g["r4", n] * x9 / 1000
                )
            )
            - 1
        )
        <= -g["r4", n]
    )

    e4[n] = (
        g["r5", n] * x1
        - x10
        - (1 - x1 * x2)
        * x4
        * (
            gams_math.exp(
                x6
                * (
                    g["r1", n]
                    - g["r2", n]
                    - g["r3", n] * x7 / 1000
                    + g["r4", n] * x9 / 1000
                )
            )
            - 1
        )
        <= g["r4", n]
    )

    e[...] = x1 * x3 - x2 * x4 == 0

    eobj[...] = obj == x10

    # Bounds on variables
    x1.lo[...] = 0
    x1.up[...] = 10
    x2.lo[...] = 0
    x2.up[...] = 10
    x3.lo[...] = 0
    x3.up[...] = 10
    x4.lo[...] = 0
    x4.up[...] = 10
    x5.lo[...] = 0
    x5.up[...] = 10
    x6.lo[...] = 0
    x6.up[...] = 10
    x7.lo[...] = 0
    x7.up[...] = 10
    x8.lo[...] = 0
    x8.up[...] = 10
    x9.lo[...] = 0
    x9.up[...] = 10
    x10.lo[...] = 0
    x10.up[...] = 10

    # Initial point
    x1.l[...] = 0.7
    x2.l[...] = 0.38
    x3.l[...] = 0.8
    x4.l[...] = 1.5
    x5.l[...] = 6
    x6.l[...] = 6
    x7.l[...] = 4
    x8.l[...] = 1
    x9.l[...] = 1.6
    x10.l[...] = 1

    circuit = Model(
        cont,
        name="circuit",
        equations=cont.getEquations(),
        problem="nlp",
        sense="min",
        objective=obj,
    )

    circuit.solve()

    import math

    assert math.isclose(circuit.objective_value, 0, rel_tol=0.001)


if __name__ == "__main__":
    main()
