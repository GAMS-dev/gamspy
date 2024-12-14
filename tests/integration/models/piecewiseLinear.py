"""
## LICENSETYPE: Requires license
## MODELTYPE: MIP
## KEYWORDS: piecewise linear function, binary, sos2


Piecewise Linear
----------------

Description: A set of models for testing Piecewise Linear function implementation

Usage: python piecewiseLinear.py
"""

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
                assert y.toDense() == expected_y
                assert x.toDense() == expected_x

        print(f"Case {case_i} passed !")


if __name__ == "__main__":
    main()
