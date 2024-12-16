"""
## LICENSETYPE: Requires license
## MODELTYPE: MIP
## KEYWORDS: piecewise linear function, binary, sos2


Piecewise Linear
----------------

Description: A set of models for testing Piecewise Linear function implementation

Usage: python piecewiseLinear.py
"""

import math

import numpy as np

import gamspy as gp


def main():
    print("Piecewise linear function test model")
    m = gp.Container()
    x = gp.Variable(m, name="x")

    np.random.seed(1997)
    x_points_1 = [
        int(x)
        for x in sorted(np.random.randint(low=-1000, high=1000, size=(1000)))
    ]
    y_points_1 = [
        int(x) for x in (np.random.randint(low=-1000, high=1000, size=(1000)))
    ]
    xy = list(zip(x_points_1, y_points_1))
    max_pair = max(xy, key=lambda k: k[1])
    min_pair = min(xy, key=lambda k: k[1])

    # A line segment between -1 and 1
    test_cases = [
        ([-1, 1], [-5, 5], -5, 5, -1, 1),
        ([-1.1, 1, 100], [-5.2, 5, 20], -5.2, 20, -1.1, 100),
        ([-1, 1, 1], [5, -5, -10], -10, 5, 1, -1),
        (
            [-1, -1, 1],
            [5, -5, 0],
            -5,
            5,
            -1,
            -1,
        ),
        (
            x_points_1,
            y_points_1,
            min_pair[1],
            max_pair[1],
            min_pair[0],
            max_pair[0],
        ),
    ]

    for case_i, (
        x_points,
        y_points,
        exp_min,
        exp_max,
        x_at_min,
        x_at_max,
    ) in enumerate(test_cases):
        for sense, expected_y, expected_x in [
            ("min", exp_min, x_at_min),
            ("max", exp_max, x_at_max),
        ]:
            for using in ["sos2", "binary"]:
                y, eqs = gp.formulations.piecewise_linear_function(
                    x,
                    x_points,
                    y_points,
                    using=using,
                )
                model = gp.Model(
                    m, equations=eqs, objective=y, sense=sense, problem="mip"
                )
                model.solve()
                assert y.toDense() == expected_y, f"Case {case_i} failed !"
                assert x.toDense() == expected_x, f"Case {case_i} failed !"

        print(f"Case {case_i} passed !")

    # test bound cases
    # y is not bounded
    x_points = [-4, -2, 1, 3]
    y_points = [-2, 0, 0, 2]
    y, eqs = gp.formulations.piecewise_linear_function(
        x, x_points, y_points, bound_domain=False
    )
    x.fx[...] = -5
    model = gp.Model(m, equations=eqs, objective=y, sense="min", problem="mip")
    model.solve()

    assert math.isclose(y.toDense(), -3), "Case 5 failed !"
    print("Case 5 passed !")
    x.fx[...] = 100
    model.solve()
    assert math.isclose(y.toDense(), 99), "Case 6 failed !"
    print("Case 6 passed !")

    # y is upper bounded
    x_points = [-4, -2, 1, 3]
    y_points = [-2, 0, 0, 0]
    y, eqs = gp.formulations.piecewise_linear_function(
        x, x_points, y_points, bound_domain=False
    )
    model = gp.Model(m, equations=eqs, objective=y, sense="max", problem="mip")
    model.solve()
    assert math.isclose(y.toDense(), 0), "Case 7 failed !"
    print("Case 7 passed !")
    x.fx[...] = 100
    model.solve()
    assert math.isclose(y.toDense(), 0), "Case 8 failed !"
    print("Case 8 passed !")

    # y is lower bounded
    x_points = [-4, -2, 1, 3]
    y_points = [-5, -5, 0, 2]
    y, eqs = gp.formulations.piecewise_linear_function(
        x, x_points, y_points, bound_domain=False
    )
    x.lo[...] = "-inf"
    x.up[...] = "inf"
    model = gp.Model(m, equations=eqs, objective=y, sense="min", problem="mip")
    model.solve()
    assert math.isclose(y.toDense(), -5), "Case 9 failed !"
    print("Case 9 passed !")
    x.fx[...] = -100
    model.solve()
    assert math.isclose(y.toDense(), -5), "Case 10 failed !"
    print("Case 10 passed !")

    # test discontinuous function not allowing in between value
    x_points = [1, 4, 4, 10]
    y_points = [1, 4, 8, 25]
    y, eqs = gp.formulations.piecewise_linear_function(
        x, x_points, y_points, bound_domain=True
    )
    x.fx[...] = 4
    y.fx[...] = 6
    model = gp.Model(m, equations=eqs, objective=y, sense="max", problem="mip")
    res = model.solve()
    assert (
        res["Model Status"].item() == "IntegerInfeasible"
    ), "Case 11 failed !"
    print("Case 11 passed !")


if __name__ == "__main__":
    main()
