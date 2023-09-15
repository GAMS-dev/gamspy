"""
Economic load dispatch for 15 generator systems with transmission losses
modeled using B-matrix formulation (Kron).
EDC of a total power of 1980 MW using 15 power generating units.
"""
import numpy as np

import gamspy.math as gams_math
from gamspy import Alias
from gamspy import Container
from gamspy import Equation
from gamspy import Model
from gamspy import Parameter
from gamspy import Set
from gamspy import Sum
from gamspy import Variable


def data_records():
    # bound records table
    bound_recs = np.array(
        [
            [100, 655],
            [100, 455],
            [20, 130],
            [20, 130],
            [150, 470],
            [135, 460],
            [135, 465],
            [100, 300],
            [25, 165],
            [25, 460],
            [20, 80],
            [20, 80],
            [25, 85],
            [15, 55],
            [15, 55],
        ]
    )

    # data records table
    data_recs = np.array(
        [
            [0.000299, 10.100, 671.130],
            [0.000183, 10.200, 574.010],
            [0.001126, 8.814, 374.110],
            [0.001126, 8.800, 374.000],
            [0.000205, 10.400, 461.000],
            [0.000301, 10.100, 630.000],
            [0.000364, 9.800, 548.000],
            [0.000338, 11.200, 227.000],
            [0.000807, 11.200, 173.000],
            [0.001203, 10.700, 175.200],
            [0.003586, 10.200, 186.000],
            [0.005513, 9.900, 230.000],
            [0.000371, 13.100, 225.000],
            [0.001929, 12.100, 309.000],
            [0.004447, 12.400, 323.100],
        ]
    )

    # Losscoef records table
    Losscoef_recs = np.array(
        [
            [
                1.4,
                1.2,
                0.7,
                0.1,
                0.3,
                0.1,
                0.1,
                0.1,
                0.3,
                0.5,
                0.3,
                0.2,
                0.4,
                0.3,
                0.1,
            ],
            [
                1.2,
                1.5,
                1.3,
                0.0,
                0.5,
                0.2,
                0.0,
                0.1,
                0.2,
                0.4,
                0.4,
                0.0,
                0.4,
                1.0,
                0.2,
            ],
            [
                0.7,
                1.3,
                7.6,
                0.1,
                1.3,
                0.9,
                0.1,
                0.0,
                0.8,
                1.2,
                1.7,
                0.0,
                2.6,
                11.1,
                2.8,
            ],
            [
                0.1,
                0.0,
                0.1,
                3.4,
                0.7,
                0.4,
                1.1,
                5.0,
                2.9,
                3.2,
                1.1,
                0.0,
                0.1,
                0.1,
                2.6,
            ],
            [
                0.3,
                0.5,
                1.3,
                0.7,
                9.0,
                1.4,
                0.3,
                1.2,
                1.0,
                1.3,
                0.7,
                0.2,
                0.2,
                2.4,
                0.3,
            ],
            [
                0.1,
                0.2,
                0.9,
                0.4,
                1.4,
                1.6,
                0.0,
                0.6,
                0.5,
                0.8,
                1.1,
                0.1,
                0.2,
                1.7,
                0.3,
            ],
            [
                0.1,
                0.0,
                0.1,
                1.1,
                0.3,
                0.0,
                1.5,
                1.7,
                1.5,
                0.9,
                0.5,
                0.7,
                0.0,
                0.2,
                0.8,
            ],
            [
                0.1,
                0.1,
                0.0,
                5.0,
                1.2,
                0.6,
                1.7,
                16.8,
                8.2,
                7.9,
                2.3,
                3.6,
                0.1,
                0.5,
                7.8,
            ],
            [
                0.3,
                0.2,
                0.8,
                2.9,
                1.0,
                0.5,
                1.5,
                8.2,
                12.9,
                11.6,
                2.1,
                2.5,
                0.7,
                1.2,
                7.2,
            ],
            [
                0.5,
                0.4,
                1.2,
                3.2,
                1.3,
                0.8,
                0.9,
                7.9,
                11.6,
                20.0,
                2.7,
                3.4,
                0.9,
                1.1,
                8.8,
            ],
            [
                0.3,
                0.4,
                1.7,
                1.1,
                0.7,
                1.1,
                0.5,
                2.3,
                2.1,
                2.7,
                14.0,
                0.1,
                0.4,
                3.8,
                16.8,
            ],
            [
                0.2,
                0.0,
                0.0,
                0.0,
                0.2,
                0.1,
                0.7,
                3.6,
                2.5,
                3.4,
                0.1,
                5.4,
                0.1,
                0.4,
                2.8,
            ],
            [
                0.4,
                0.4,
                2.6,
                0.1,
                0.2,
                0.2,
                0.0,
                0.1,
                0.7,
                0.9,
                0.4,
                0.1,
                10.3,
                10.1,
                2.8,
            ],
            [
                0.3,
                1.0,
                11.1,
                0.1,
                2.4,
                1.7,
                0.2,
                0.5,
                1.2,
                1.1,
                3.8,
                0.4,
                10.1,
                57.8,
                9.4,
            ],
            [
                0.1,
                0.2,
                2.8,
                2.6,
                0.3,
                0.3,
                0.8,
                7.8,
                7.2,
                8.8,
                16.8,
                2.8,
                2.8,
                9.4,
                128.3,
            ],
        ]
    )

    return bound_recs, data_recs, Losscoef_recs


def main():
    m = Container()

    # SETS #
    i = Set(
        m,
        name="i",
        records=[str(i) for i in range(1, 16)],
        description="generating units",
    )
    bou = Set(
        m, name="bou", records=["low", "upp"], description="lower and upper"
    )
    coef = Set(
        m,
        name="coef",
        records=["a", "b", "c"],
        description="coefficients in fuel cost of thermal generating unit",
    )

    # ALIAS #
    j = Alias(m, name="j", alias_with=i)

    # PARAMETERS #

    # The output of the minimum and maximum operation of the
    # generating units in MW.
    bound = Parameter(
        m, name="bound", domain=[i, bou], records=data_records()[0]
    )

    # The cost coefficients of generator units.
    data = Parameter(
        m, name="data", domain=[i, coef], records=data_records()[1]
    )

    # The loss coefficients
    Losscoef = Parameter(
        m, name="Losscoef", domain=[i, j], records=data_records()[2]
    )

    Load = Parameter(m, name="Load", records=1980)

    # VARIABLES #
    P = Variable(
        m, name="P", domain=[i], description="optimal generation level of i"
    )
    obj = Variable(m, name="obj", description="minimum cost")

    # EQUATIONS #
    cost = Equation(
        m, name="cost", type="regular", description="total generation cost"
    )
    bal = Equation(
        m, name="bal", type="regular", description="demand-supply balance"
    )

    # Objective function:
    cost.expr = obj == Sum(
        i,
        data[i, "a"] * gams_math.power(P[i], 2)
        + data[i, "b"] * P[i]
        + data[i, "c"],
    )

    # Constraints:
    bal.expr = (
        Sum(i, P[i]) - Sum([i, j], P[i] * Losscoef[i, j] * P[j] / 10000)
        == Load
    )

    # Bounds on variables:
    P.lo[i] = bound[i, "low"]
    P.up[i] = bound[i, "upp"]

    P.l[i] = (bound[i, "low"] + bound[i, "upp"]) / 2

    edc2 = Model(
        m,
        name="edc2",
        equations=m.getEquations(),
        problem="nlp",
        sense="MIN",
        objective=obj,
    )

    edc2.solve()

    print("Objective Function Value:  ", round(obj.toValue(), 4))
    # End edc2


if __name__ == "__main__":
    main()
