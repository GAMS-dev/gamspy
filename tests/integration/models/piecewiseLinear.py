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
import gamspy.formulations.piecewise as piecewise


def pwl_suite(fct, name):
    print(f"PWL Suite function: {name}")
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
            if name == "convexity":
                for using in ["sos2", "binary"]:
                    y, eqs = fct(
                        x,
                        x_points,
                        y_points,
                        using=using,
                    )
                    model = gp.Model(
                        m,
                        equations=eqs,
                        objective=y,
                        sense=sense,
                        problem="mip",
                    )
                    model.solve()
                    assert y.toDense() == expected_y, f"Case {case_i} failed !"
                    assert x.toDense() == expected_x, f"Case {case_i} failed !"
            else:
                y, eqs = fct(
                    x,
                    x_points,
                    y_points,
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
    if name == "convexity":
        for using in ["sos2", "binary"]:
            x_points = [-4, -2, 1, 3]
            y_points = [-2, 0, 0, 2]
            y, eqs = fct(
                x,
                x_points,
                y_points,
                using=using,
                bound_left=False,
                bound_right=False,
            )
            x.fx = -5
            model = gp.Model(
                m, equations=eqs, objective=y, sense="min", problem="mip"
            )
            model.solve()
            assert math.isclose(y.toDense(), -3), "Case 5 failed !"
            print("Case 5 passed !")
            x.fx = 100
            model.solve()
            assert math.isclose(y.toDense(), 99), "Case 6 failed !"
            print("Case 6 passed !")

            # y is upper bounded
            x_points = [-4, -2, 1, 3]
            y_points = [-2, 0, 0, 0]
            y, eqs = fct(
                x,
                x_points,
                y_points,
                using=using,
                bound_left=False,
                bound_right=False,
            )
            model = gp.Model(
                m, equations=eqs, objective=y, sense="max", problem="mip"
            )
            model.solve()
            assert math.isclose(y.toDense(), 0), "Case 7 failed !"
            print("Case 7 passed !")
            x.fx = 100
            model.solve()
            assert math.isclose(y.toDense(), 0), "Case 8 failed !"
            print("Case 8 passed !")
            # y is lower bounded
            x_points = [-4, -2, 1, 3]
            y_points = [-5, -5, 0, 2]
            y, eqs = fct(
                x,
                x_points,
                y_points,
                using=using,
                bound_left=False,
                bound_right=False,
            )
            x.lo = gp.SpecialValues.NEGINF
            x.up = gp.SpecialValues.POSINF
            model = gp.Model(
                m, equations=eqs, objective=y, sense="min", problem="mip"
            )
            model.solve()
            assert math.isclose(y.toDense(), -5), "Case 9 failed !"
            print("Case 9 passed !")
            x.fx = -100
            model.solve()
            assert math.isclose(y.toDense(), -5), "Case 10 failed !"
            print("Case 10 passed !")
    else:
        x_points = [-4, -2, 1, 3]
        y_points = [-2, 0, 0, 2]
        y, eqs = fct(
            x, x_points, y_points, bound_left=False, bound_right=False
        )
        x.fx = -5
        model = gp.Model(
            m, equations=eqs, objective=y, sense="min", problem="mip"
        )
        model.solve()
        assert math.isclose(y.toDense(), -3), "Case 5 failed !"
        print("Case 5 passed !")
        x.fx = 100
        model.solve()
        assert math.isclose(y.toDense(), 99), "Case 6 failed !"
        print("Case 6 passed !")

        # y is upper bounded
        x_points = [-4, -2, 1, 3]
        y_points = [-2, 0, 0, 0]
        y, eqs = fct(
            x, x_points, y_points, bound_left=False, bound_right=False
        )
        model = gp.Model(
            m, equations=eqs, objective=y, sense="max", problem="mip"
        )
        model.solve()
        assert math.isclose(y.toDense(), 0), "Case 7 failed !"
        print("Case 7 passed !")
        x.fx = 100
        model.solve()
        assert math.isclose(y.toDense(), 0), "Case 8 failed !"
        print("Case 8 passed !")
        # y is lower bounded
        x_points = [-4, -2, 1, 3]
        y_points = [-5, -5, 0, 2]
        y, eqs = fct(
            x, x_points, y_points, bound_left=False, bound_right=False
        )
        x.lo = gp.SpecialValues.NEGINF
        x.up = gp.SpecialValues.POSINF
        model = gp.Model(
            m, equations=eqs, objective=y, sense="min", problem="mip"
        )
        model.solve()
        assert math.isclose(y.toDense(), -5), "Case 9 failed !"
        print("Case 9 passed !")
        x.fx = -100
        model.solve()
        assert math.isclose(y.toDense(), -5), "Case 10 failed !"
        print("Case 10 passed !")

    # test discontinuous function not allowing in between value
    x_points = [1, 4, 4, 10]
    y_points = [1, 4, 8, 25]
    y, eqs = fct(x, x_points, y_points)
    x.fx = 4
    y.fx = 6  # y can be either 4 or 8 but not their convex combination
    model = gp.Model(m, equations=eqs, objective=y, sense="max", problem="mip")
    res = model.solve()
    assert (
        res["Model Status"].item() == "IntegerInfeasible"
    ), "Case 11 failed !"
    print("Case 11 passed !")

    # test None case
    x_points = [1, 4, None, 6, 10]
    y_points = [1, 4, None, 8, 25]
    y, eqs = fct(x, x_points, y_points)
    x.fx = 5  # should be IntegerInfeasible since 5 \in [4, 6]
    model = gp.Model(m, equations=eqs, objective=y, sense="max", problem="mip")
    res = model.solve()
    assert (
        res["Model Status"].item() == "IntegerInfeasible"
    ), "Case 12 failed !"
    print("Case 12 passed !")

    # test None case
    x_points = [1, 4, None, 6, 10]
    y_points = [1, 4, None, 30, 25]
    y, eqs = fct(x, x_points, y_points)
    x.lo = gp.SpecialValues.NEGINF
    x.up = gp.SpecialValues.POSINF
    model = gp.Model(m, equations=eqs, objective=y, sense="max", problem="mip")
    res = model.solve()
    assert x.toDense() == 6, "Case 13 failed !"
    assert y.toDense() == 30, "Case 13 failed !"
    print("Case 13 passed !")

    # test None case
    x_points = [1, 4, None, 6, 10]
    y_points = [1, 45, None, 30, 25]
    y, eqs = fct(x, x_points, y_points)
    x.lo = gp.SpecialValues.NEGINF
    x.up = gp.SpecialValues.POSINF
    model = gp.Model(m, equations=eqs, objective=y, sense="max", problem="mip")
    res = model.solve()
    assert x.toDense() == 4, "Case 14 failed !"
    assert y.toDense() == 45, "Case 14 failed !"
    print("Case 14 passed !")

    # test with a non-scalar input
    i = gp.Set(m, name="i", records=["1", "2", "3", "4", "5"])
    x2 = gp.Variable(m, name="x2", domain=[i])
    x_points = [1, 4, None, 6, 10, 10, 20]
    y_points = [1, 45, None, 30, 25, 30, 12]
    y, eqs = fct(x2, x_points, y_points)
    x2.fx["1"] = 1
    x2.fx["2"] = 2.5
    x2.fx["3"] = 8
    x2.fx["4"] = 4
    x2.fx["5"] = 15
    model = gp.Model(
        m,
        equations=eqs,
        objective=gp.Sum(y.domain, y),
        sense="max",
        problem="mip",
    )
    model.solve()
    assert np.allclose(
        y.toDense(), np.array([1, 23, 27.5, 45, 21])
    ), "Case 14 failed !"
    print("Case 14 passed !")

    # test unbounded when edges are discontinuous
    if name == "convexity":
        for using in ["binary", "sos2"]:
            x_points = [-4, -4, -2, 1, 3, 3]
            y_points = [20, -2, 0, 0, 2, 9]
            y, eqs = fct(
                x,
                x_points,
                y_points,
                using=using,
                bound_left=False,
                bound_right=False,
            )
            x.fx = -5
            model = gp.Model(
                m, equations=eqs, objective=y, sense="min", problem="mip"
            )
            model.solve()
            assert y.toDense() == 20, "Case 15 failed !"

        print("Case 15 passed !")
    else:
        x_points = [-4, -4, -2, 1, 3, 3]
        y_points = [20, -2, 0, 0, 2, 9]
        y, eqs = fct(
            x, x_points, y_points, bound_left=False, bound_right=False
        )
        x.fx = -5
        model = gp.Model(
            m, equations=eqs, objective=y, sense="min", problem="mip"
        )
        model.solve()
        assert y.toDense() == 20, "Case 15 failed !"
        print("Case 15 passed !")

    # test single bound cases
    x_points = [-4, -2, 1, 3]
    y_points = [-2, 0, 0, 2]
    y, eqs = fct(
        x,
        x_points,
        y_points,
        bound_left=False,
        bound_right=True,
    )
    x.fx = -5
    model = gp.Model(m, equations=eqs, objective=y, sense="min", problem="mip")
    model.solve()

    assert y.toDense() == -3, "Case 16 failed !"
    print("Case 16 passed !")

    x.fx = 5  # bounded from right
    res = model.solve()
    assert (
        res["Model Status"].item() == "IntegerInfeasible"
    ), "Case 17 failed !"
    print("Case 17 passed !")

    y, eqs = fct(
        x,
        x_points,
        y_points,
        bound_left=True,
        bound_right=False,
    )
    x.fx = -5
    model = gp.Model(m, equations=eqs, objective=y, sense="min", problem="mip")
    res = model.solve()
    assert (
        res["Model Status"].item() == "IntegerInfeasible"
    ), "Case 18 failed !"
    print("Case 18 passed !")

    x.fx = 5
    model.solve()

    assert y.toDense() == 4, "Case 19 failed !"
    print("Case 19 passed !")

    m.close()


def indicator_suite():
    print("Indicator Suite")
    m = gp.Container()
    x = gp.Variable(m, name="x")
    b = gp.Variable(m, name="b", type="binary")

    eqs1 = piecewise._indicator(b, 1, x <= 50)
    model = gp.Model(
        m,
        equations=eqs1,
        objective=x,
        sense="max",
        problem="mip",
    )
    b.fx = 1
    model.solve()
    assert x.toDense() == 50, "Case 1 failed !"
    print("Case 1 passed !")

    eqs2 = piecewise._indicator(b, 0, x <= 500)
    b.lo = 0
    b.up = 1

    model = gp.Model(
        m,
        equations=[*eqs1, *eqs2],
        objective=x,
        sense="max",
        problem="mip",
    )
    model.solve()

    assert x.toDense() == 500, "Case 2 failed !"
    assert b.toDense() == 0, "Case 2 failed !"
    print("Case 2 passed !")

    eqs3 = piecewise._indicator(b, 1, x <= 50)
    model = gp.Model(
        m,
        equations=eqs3,
        objective=x,
        sense="max",
        problem="mip",
    )
    # b = 0 does not mean x cannot be less than 50
    b.fx = 0
    x.fx = 20
    model.solve()
    assert x.toDense() == 20, "Case 3 failed !"
    print("Case 3 passed !")

    eqs4 = piecewise._indicator(b, 1, x == 120)
    model = gp.Model(
        m,
        equations=eqs4,
        objective=x,
        sense="min",
        problem="mip",
    )
    b.fx = 1
    x.lo = gp.SpecialValues.NEGINF
    x.up = gp.SpecialValues.POSINF
    model.solve()
    assert x.toDense() == 120, "Case 4 failed !"
    print("Case 4 passed !")

    eqs5 = piecewise._indicator(b, 1, x >= 99)
    model = gp.Model(
        m,
        equations=eqs5,
        objective=x,
        sense="min",
        problem="mip",
    )
    b.fx = 1
    x.lo = gp.SpecialValues.NEGINF
    x.up = gp.SpecialValues.POSINF
    model.solve()
    assert x.toDense() == 99, "Case 5 failed !"
    print("Case 5 passed !")
    m.close()


if __name__ == "__main__":
    print("Piecewise linear function test model")
    pwl_suite(piecewise.pwl_convexity_formulation, "convexity")
    pwl_suite(piecewise.pwl_interval_formulation, "interval")
    indicator_suite()
