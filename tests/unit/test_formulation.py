from __future__ import annotations

import math

import numpy as np
import pytest

import gamspy as gp
import gamspy.formulations.piecewise as piecewise
from gamspy.exceptions import ValidationError
from gamspy.formulations import (
    PWLCurve,
    pwl_convexity_formulation,
    pwl_dlog_formulation,
    pwl_interval_formulation,
    pwlinear,
)

pytestmark = pytest.mark.unit

fcts_to_test = [
    pwl_convexity_formulation,
    pwl_dlog_formulation,
    pwl_interval_formulation,
]


@pytest.fixture
def data():
    m = gp.Container()
    x = gp.Variable(m, "x")
    x2 = gp.Variable(m, "x2", domain=gp.math.dim([2, 4, 3]))
    x_points = [-10, 2.2, 5, 10]
    y_points = [10, 20, -2, -5]
    return {
        "m": m,
        "x": x,
        "x2": x2,
        "x_points": x_points,
        "y_points": y_points,
    }


def get_var_count_by_type(m: gp.Container) -> dict[str, int]:
    count = {}
    for k in m.data:
        symbol = m.data[k]
        if not isinstance(symbol, gp.Variable):
            continue

        sym_type = symbol.type
        if sym_type not in count:
            count[sym_type] = 0

        count[sym_type] += 1

    return count


def test_pwl_enforce_sos2_log_binary():
    m = gp.Container()
    i = gp.Set(m, name="i", records=["1", "2", "3"])
    lambda_var = gp.Variable(m, name="lambda", domain=[i])
    # this will create binary variables
    eqs = piecewise._enforce_sos2_with_binary(lambda_var)
    assert len(eqs) == 2
    var_count = get_var_count_by_type(m)
    assert var_count["binary"] == 1


def test_pwl_enforce_sos2_log_binary_2():
    m = gp.Container()
    i = gp.Set(m, name="i", records=["1", "2"])
    lambda_var = gp.Variable(m, name="lambda", domain=[i])
    # this will create binary variables
    eqs = piecewise._enforce_sos2_with_binary(lambda_var)
    assert len(eqs) == 0
    var_count = get_var_count_by_type(m)
    assert "binary" not in var_count


def test_pwl_indicator():
    m = gp.Container()
    i = gp.Set(m, name="i", records=["1", "2"])
    j = gp.Set(m, name="j", records=["1", "2", "3"])
    k = gp.Set(m, name="k", records=["a", "b"])

    b = gp.Variable(m, name="b", type="binary", domain=[i])
    b2 = gp.Variable(m, name="b2", type="free", domain=[j])
    x = gp.Variable(m, name="x", domain=[i])
    x2 = gp.Variable(m, name="x2", domain=[j])
    x3 = gp.Variable(m, name="x3", domain=[k])
    x4 = gp.Variable(m, name="x4", domain=[i, k])

    b3 = gp.Variable(m, name="b3", type="binary")
    x5 = gp.Variable(m, name="x5")

    with pytest.raises(ValidationError):
        piecewise._indicator("indicator_var", 0, x <= 10)

    with pytest.raises(ValidationError):
        piecewise._indicator(b2, 0, x <= 10)
    with pytest.raises(ValidationError):
        piecewise._indicator(b, -1, x <= 10)
    with pytest.raises(ValidationError):
        piecewise._indicator(b, 0, x)
    with pytest.raises(ValidationError):
        piecewise._indicator(b, 0, x + 10)
    with pytest.raises(ValidationError):
        piecewise._indicator(b, 0, x3 >= 10)
    with pytest.raises(ValidationError):
        piecewise._indicator(b, 0, x2 >= 10)
    with pytest.raises(ValidationError):
        piecewise._indicator(b, 0, x4 >= 10)

    eqs1 = piecewise._indicator(b, 0, x >= 10)
    eqs2 = piecewise._indicator(b, 0, x <= 10)
    eqs3 = piecewise._indicator(b, 0, x == 10)
    assert len(eqs1) == len(eqs2)
    assert len(eqs3) == len(eqs1) * 2

    eqs4 = piecewise._indicator(b3, 1, x5 >= 10)
    assert len(eqs4) == len(eqs1)

    var_count = get_var_count_by_type(m)
    assert "sos1" in var_count

    piecewise._indicator(b, 1, x >= 10)
    piecewise._indicator(b, 1, x <= 10)
    piecewise._indicator(b, 1, x == 10)


def test_pwl_enforce_sos2_log_binary_with_domain():
    m = gp.Container()
    j = gp.Set(m, name="j", records=["1", "2"])
    i = gp.Set(m, name="i", records=["1", "2", "3"])
    lambda_var = gp.Variable(m, name="lambda", domain=[j, i])
    # this will create binary variables
    eqs = piecewise._enforce_sos2_with_binary(lambda_var)
    assert len(eqs) == 2
    var_count = get_var_count_by_type(m)
    assert var_count["binary"] == 1

    for k in m.data:
        sym = m.data[k]
        if isinstance(sym, gp.Equation):
            assert len(sym.domain) == 2
            assert sym.domain[0] == j


def test_pwl_enforce_sos2_log_binary_with_domain_2():
    m = gp.Container()
    lambda_var = gp.Variable(m, name="lambda", domain=gp.math.dim([3, 8]))
    # this will create binary variables
    eqs = piecewise._enforce_sos2_with_binary(lambda_var)
    assert len(eqs) == 2
    var_count = get_var_count_by_type(m)
    assert var_count["binary"] == 1

    for k in m.data:
        sym = m.data[k]
        if isinstance(sym, gp.Equation):
            assert len(sym.domain) == 2
            print(sym.getDefinition())


def test_pwl_enforce_discontinuity():
    m = gp.Container()
    lambda_var = gp.Variable(m, name="lambda", domain=gp.math.dim([5, 5]))
    # this will create binary variables
    eqs = piecewise._enforce_discontinuity(lambda_var, [1, 3])
    assert len(eqs) == 2
    assert len(eqs[0].domain) == 4
    assert len(eqs[1].domain) == 4


def test_pwl_gray_code():
    for n, m in [(2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (513, 10), (700, 10)]:
        code = piecewise._generate_gray_code(n, m)
        old = None
        for row in code:
            if old is None:
                old = row
                continue

            diff = old - row
            count = 0
            for col in diff:
                count += abs(col)

            # in gray code consecutive two rows differ by 1 bit
            assert count == 1, "Gray code row had more than 1 change"
            old = row


def test_pwl_dlog_structure():
    for segment_count in [1, 2, 3, 5]:
        m = gp.Container()
        input_x = gp.Variable(m, "input_x")
        x_points = list(range(segment_count + 1))
        y_points = [point**2 for point in x_points]
        variables_before = set(m.getVariables())

        _output_y, equations = pwl_dlog_formulation(
            input_x,
            x_points,
            y_points,
        )

        new_variables = set(m.getVariables()) - variables_before
        gamma_variables = [
            variable for variable in new_variables if variable.type == "positive"
        ]
        binary_variables = [
            variable for variable in new_variables if variable.type == "binary"
        ]
        bit_count = math.ceil(math.log2(segment_count)) if segment_count > 1 else 0
        expected_binary_count = 1 if bit_count else 0

        assert len(gamma_variables) == 2
        assert all(
            len(variable.domain[-1]) == segment_count for variable in gamma_variables
        )
        assert len(binary_variables) == expected_binary_count
        assert len(equations) == 3 + expected_binary_count
        if bit_count:
            assert len(binary_variables[0].domain[-1]) == bit_count


def test_pwl_calling_conventions():
    m = gp.Container()
    x = gp.Variable(m, "x")
    x_points = [-1, 1, 3]
    y_points = [-5, 5, 9]

    y_interval, eqs_interval = pwl_interval_formulation(x, x_points, y_points)
    y_convexity, eqs_convexity = pwl_convexity_formulation(x, x_points, y_points)
    y_sos2, eqs_sos2 = pwl_convexity_formulation(x, x_points, y_points, "sos2")

    assert y_interval.lo.toDense() == -5
    assert y_interval.up.toDense() == 9
    assert y_convexity.lo.toDense() == -5
    assert y_convexity.up.toDense() == 9
    assert y_sos2.lo.toDense() == -5
    assert y_sos2.up.toDense() == 9
    assert eqs_interval
    assert eqs_convexity
    assert eqs_sos2
    var_count = get_var_count_by_type(m)
    assert var_count["binary"] == 2
    assert var_count["sos2"] == 1

    with pytest.raises(TypeError):
        pwl_interval_formulation(x, x_points, y_points, False)

    with pytest.raises(TypeError):
        pwl_convexity_formulation(x, x_points, y_points, "binary", False)

    with pytest.raises(TypeError):
        pwl_dlog_formulation(x, x_points, y_points, False)


@pytest.mark.parametrize("fct", fcts_to_test)
@pytest.mark.parametrize("indexed", [False, True], ids=["scalar", "indexed"])
def test_pwl_tuple_input_and_return(fct, indexed):
    m = gp.Container()
    if indexed:
        i = gp.Set(m, "i", records=["i1", "i2"])
        j = gp.Set(m, "j", records=["j1", "j2", "j3"])
        x = gp.Variable(m, "x", domain=[i, j])
    else:
        x = gp.Variable(m, "x")

    result = fct(x, (-1, 1, 3), (-5, 5, 9))

    assert isinstance(result, tuple)
    output_y, equations = result
    assert isinstance(output_y, gp.Variable)
    assert output_y.domain == x.domain
    assert isinstance(equations, list)
    assert equations
    assert all(isinstance(equation, gp.Equation) for equation in equations)


@pytest.mark.parametrize("fct", fcts_to_test)
@pytest.mark.parametrize(
    "bound_left, bound_right, expected_lower, expected_upper",
    [
        (True, True, -2.0, 2.0),
        (False, True, gp.SpecialValues.NEGINF, 2.0),
        (True, False, -2.0, gp.SpecialValues.POSINF),
        (
            False,
            False,
            gp.SpecialValues.NEGINF,
            gp.SpecialValues.POSINF,
        ),
    ],
)
def test_pwl_output_bounds(
    fct,
    bound_left,
    bound_right,
    expected_lower,
    expected_upper,
):
    m = gp.Container()
    x = gp.Variable(m, "x")

    output_y, _ = fct(
        x,
        [-4, -2, 1, 3],
        [-2, 0, 0, 2],
        bound_left=bound_left,
        bound_right=bound_right,
    )

    assert output_y.lo.toDense() == expected_lower
    assert output_y.up.toDense() == expected_upper


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_decreasing_ray_output_bounds(fct):
    left_m = gp.Container()
    left_x = gp.Variable(left_m, "left_x")
    left_y, _ = fct(
        left_x,
        [-4, -2, 1, 3],
        [2, 0, 0, -2],
        bound_left=False,
    )

    assert left_y.lo.toDense() == -2
    assert left_y.up.toDense() == gp.SpecialValues.POSINF

    right_m = gp.Container()
    right_x = gp.Variable(right_m, "right_x")
    right_y, _ = fct(
        right_x,
        [-4, -2, 1, 3],
        [2, 0, 0, -2],
        bound_right=False,
    )

    assert right_y.lo.toDense() == gp.SpecialValues.NEGINF
    assert right_y.up.toDense() == 2


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_validation_does_not_add_symbols(fct):
    m = gp.Container()
    x = gp.Variable(m, "x")
    symbols_before = list(m.data.keys())

    with pytest.raises(ValidationError):
        fct(x, [0, 1, 2], [0, 1])

    assert list(m.data.keys()) == symbols_before

    empty_container = gp.Container()
    empty_domain = gp.Set(empty_container)
    empty_x = gp.Variable(empty_container, domain=empty_domain)
    empty_symbols_before = list(empty_container.data)
    with pytest.raises(ValidationError, match="domain cannot be empty"):
        fct(empty_x, [0, 1], [0, 1])
    assert list(empty_container.data) == empty_symbols_before

    wildcard_container = gp.Container()
    wildcard_x = gp.Variable(wildcard_container, domain=[gp.UNIVERSE])
    wildcard_symbols_before = list(wildcard_container.data)
    with pytest.raises(ValidationError, match="explicit Sets or Aliases"):
        fct(wildcard_x, [0, 1], [0, 1])
    assert list(wildcard_container.data) == wildcard_symbols_before


def test_pwl_with_sos2(data):
    m = data["m"]
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]
    y, _eqs = pwl_convexity_formulation(x, x_points, y_points, using="sos2")
    y2, _eqs2 = pwl_convexity_formulation(
        x,
        x_points,
        y_points,
        bound_left=False,
        bound_right=False,
        using="sos2",
    )

    # there should be no binary variables
    var_count = get_var_count_by_type(m)
    assert var_count["sos2"] == 2  # since we called it twice
    assert y.type == "free"
    assert y2.type == "free"


def test_pwl_with_binary(data):
    m = data["m"]
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]
    y, eqs = pwl_convexity_formulation(x, x_points, y_points, using="binary")
    y2, eqs2 = pwl_convexity_formulation(x, x_points, y_points, using="binary")

    # there should be no sos2 variables
    var_count = get_var_count_by_type(m)
    assert "sos2" not in var_count
    assert var_count["binary"] == 2  # since we called it twice
    assert y.type == "free"
    assert y2.type == "free"
    assert len(eqs) == len(eqs2)


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_with_domain(data, fct):
    x2 = data["x2"]
    x_points = data["x_points"]
    y_points = data["y_points"]
    y, _equations = fct(x2, x_points, y_points)

    assert len(y.domain) == len(x2.domain)

    if fct is pwl_convexity_formulation:
        sos2_y, _sos2_equations = fct(
            x2,
            x_points,
            y_points,
            using="sos2",
        )
        assert len(sos2_y.domain) == len(x2.domain)


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_with_none(data, fct):
    x = data["x"]
    x_points = [1, None, 2, 3]
    y_points = [10, None, 20, 45]
    fct(x, x_points, y_points)


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_finished_start_with_disc(data, fct):
    x = data["x"]
    x_points = [1, 1, None, 2, 3, 3]
    y_points = [0, 10, None, 20, 45, 0]
    fct(
        x,
        x_points,
        y_points,
        bound_left=False,
        bound_right=False,
        allow_multivalued=True,
    )


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_bound_cases(data, fct):
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]

    fct(x, x_points, y_points, bound_left=False, bound_right=False)
    fct(x, x_points, y_points, bound_left=False, bound_right=True)
    fct(x, x_points, y_points, bound_left=True, bound_right=True)
    fct(x, x_points, y_points, bound_left=True, bound_right=False)


@pytest.mark.parametrize("fct", fcts_to_test)
def test_pwl_validation(data, fct):
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]

    # incorrect using value
    with pytest.raises(ValidationError):
        fct(x, x_points, y_points, bound_left="hello")

    # x not a variable
    with pytest.raises(ValidationError):
        fct(10, x_points, y_points)

    # incorrect x_points, y_points
    with pytest.raises(ValidationError):
        fct(x, 10, y_points)

    with pytest.raises(ValidationError):
        fct(x, x_points, 10)

    with pytest.raises(ValidationError):
        fct(x, [1], [10])

    with pytest.raises(ValidationError):
        fct(x, x_points, [10])

    with pytest.raises(ValidationError):
        fct(x, [*x_points, "a"], [*y_points, 5])

    with pytest.raises(ValidationError):
        fct(x, [*x_points, 16], [*y_points, "a"])

    if fct is pwl_convexity_formulation:
        for decreasing_x in ([3, 2, 1], [3, 1, 2], [1, 3, 2]):
            with pytest.raises(
                ValidationError,
                match="no single non-decreasing SOS2 ordering",
            ):
                fct(
                    x,
                    decreasing_x,
                    [10, 20, 30],
                    allow_multivalued=True,
                )

    with pytest.raises(ValidationError):
        fct(x, [1], [10])

    with pytest.raises(ValidationError):
        fct(x, [None, 2, 3], [None, 20, 40])

    with pytest.raises(ValidationError):
        fct(x, [2, 3, None], [20, 40, None])

    with pytest.raises(ValidationError):
        fct(x, [None, 2, 3, None], [None, 20, 40, None])

    with pytest.raises(ValidationError):
        fct(x, [0, None, 2, 3], [0, 10, 20, 40])

    with pytest.raises(ValidationError):
        fct(x, [0, 1, 2, 3], [0, None, 20, 40])

    with pytest.raises(ValidationError):
        fct(x, [1, None, None, 2, 3], [10, None, None, 20, 40])

    with pytest.raises(ValidationError):
        fct(x, [2, None, 2, 3], [10, None, 20, 40])

    with pytest.raises(ValidationError):
        fct(x, [2, None, 4, 10], [10, None, 20, 40], bound_left="yes")

    with pytest.raises(ValidationError):
        fct(x, [2, None, 4, 10], [10, None, 20, 40], bound_right="yes")


def test_pwl_scalar_interval_normalized_input():
    m = gp.Container()
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3", "p4"])
    other_breakpoint = gp.Set(
        m, "other_breakpoint", records=["p0", "p1", "p2", "p3", "p4"]
    )
    input_x = gp.Variable(m, "input_x")
    x_points = gp.Parameter(
        m,
        "x_points",
        domain=breakpoint,
        records=[
            ("p0", 0),
            ("p1", 4),
            ("p2", gp.SpecialValues.NA),
            ("p3", 6),
            ("p4", 10),
        ],
    )
    y_points = gp.Parameter(
        m,
        "y_points",
        domain=breakpoint,
        records=[
            ("p0", 0),
            ("p1", 4),
            ("p2", gp.SpecialValues.NA),
            ("p3", 8),
            ("p4", 25),
        ],
    )
    wrong_y = gp.Parameter(
        m,
        "wrong_y",
        domain=other_breakpoint,
        records=[("p0", 0), ("p1", 4), ("p3", 8), ("p4", 25)],
    )

    output_y, equations = pwl_interval_formulation(input_x, x_points, y_points)

    assert output_y.domain == input_x.domain
    assert output_y.lo.toDense() == 0
    assert output_y.up.toDense() == 25
    assert len(equations) == 5

    symbols_before = list(m.data)
    with pytest.raises(ValidationError, match="domain must exactly match"):
        pwl_interval_formulation(input_x, x_points, wrong_y)
    assert list(m.data) == symbols_before


@pytest.mark.parametrize(
    "fct",
    [pwl_interval_formulation, pwl_dlog_formulation],
    ids=["interval", "dlog"],
)
def test_pwl_scalar_normalized_solve(fct):
    m = gp.Container()
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3"])
    input_x = gp.Variable(m, "input_x")
    x_points = gp.Parameter(
        m, "x_points", domain=breakpoint, records=np.array([0, 10, 5, 15])
    )
    y_points = gp.Parameter(
        m, "y_points", domain=breakpoint, records=np.array([0, 10, 20, 30])
    )
    output_y, equations = fct(
        input_x,
        x_points,
        y_points,
        allow_multivalued=True,
    )
    input_x.fx = 7
    model = gp.Model(
        m,
        equations=equations,
        problem="mip",
        sense="max",
        objective=output_y,
    )

    model.solve()

    assert math.isclose(output_y.toDense(), 22)

    ray_m = gp.Container()
    ray_x = gp.Variable(ray_m, "ray_x")
    ray_y, ray_equations = fct(
        ray_x,
        [-math.inf, 0, math.inf],
        [2, 1, -1],
    )
    ray_x.fx = -2
    ray_model = gp.Model(
        ray_m,
        equations=ray_equations,
        problem="mip",
        sense="min",
        objective=ray_y,
    )

    ray_model.solve()

    assert math.isclose(ray_y.toDense(), -3)

    ray_x.fx = 8
    ray_model.solve()

    assert math.isclose(ray_y.toDense(), -7)


def test_pwl_indexed_interval_normalized_input():
    m = gp.Container()
    row = gp.Set(m, "row", records=["long", "short"])
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3"])
    input_x = gp.Variable(m, "input_x", domain=row)

    broadcast_y, broadcast_equations = pwl_interval_formulation(
        input_x,
        (0, 2, 4),
        (1, 5, 9),
    )

    assert broadcast_y.domain == input_x.domain
    assert broadcast_y.lo.toDense().tolist() == [1, 1]
    assert broadcast_y.up.toDense().tolist() == [9, 9]
    assert len(broadcast_equations) == 5

    na = gp.SpecialValues.NA
    x_points = gp.Parameter(
        m,
        "x_points",
        domain=[row, breakpoint],
        records=np.array([[0, 1, 2, 3], [0, 4, na, na]]),
    )
    y_points = gp.Parameter(
        m,
        "y_points",
        domain=[row, breakpoint],
        records=np.array([[0, 2, 4, 6], [1, 9, na, na]]),
    )

    output_y, equations = pwl_interval_formulation(input_x, x_points, y_points)

    assert output_y.domain == input_x.domain
    assert output_y.lo.toDense().tolist() == [0, 1]
    assert output_y.up.toDense().tolist() == [6, 9]
    assert len(equations) == 5

    symbols_before = list(m.data)
    with pytest.raises(
        ValidationError,
        match="must both be Parameters, or both be lists or tuples",
    ):
        pwl_interval_formulation(input_x, x_points, [0, 1, 2, 3])
    assert list(m.data) == symbols_before


@pytest.mark.parametrize(
    "fct",
    [pwl_interval_formulation, pwl_dlog_formulation],
    ids=["interval", "dlog"],
)
def test_pwl_indexed_normalized_solve(fct):
    m = gp.Container()
    row = gp.Set(
        m,
        "row",
        records=["bounded", "hole", "jump", "overlap", "rays"],
    )
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3", "p4"])
    input_x = gp.Variable(m, "input_x", domain=row)
    na = gp.SpecialValues.NA
    x_points = gp.Parameter(
        m,
        "x_points",
        domain=[row, breakpoint],
        records=np.array(
            [
                [0, 1, 2, 3, 4],
                [0, 2, na, 4, 6],
                [0, 2, 2, 4, na],
                [0, 4, 2, 6, na],
                [-math.inf, 0, 2, math.inf, na],
            ]
        ),
    )
    y_points = gp.Parameter(
        m,
        "y_points",
        domain=[row, breakpoint],
        records=np.array(
            [
                [0, 1, 2, 3, 4],
                [0, 4, na, 8, 12],
                [0, 4, 6, 10, na],
                [0, 4, 5, 9, na],
                [3, 0, 4, -1, na],
            ]
        ),
    )

    variables_before = set(m.getVariables())
    output_y, equations = fct(
        input_x,
        x_points,
        y_points,
        allow_multivalued=True,
    )
    new_variables = set(m.getVariables()) - variables_before
    if fct is pwl_interval_formulation:
        # Interval uses rectangular binary selectors sized by the largest row:
        # four finite segments and two rays for this data.
        interval_selectors = [
            variable
            for variable in new_variables
            if variable.type == "binary" and variable.dimension == 2
        ]
        segment_selector = next(
            variable for variable in interval_selectors if len(variable.domain[-1]) == 4
        )
        ray_selector = next(
            variable for variable in interval_selectors if len(variable.domain[-1]) == 2
        )
    input_x.fx[row] = gp.Parameter(
        m,
        "input_level",
        domain=row,
        records=[
            ("bounded", 1.5),
            ("hole", 5),
            ("jump", 2),
            ("overlap", 3),
            ("rays", -2),
        ],
    )
    model = gp.Model(
        m,
        equations=equations,
        problem="mip",
        sense="max",
        objective=gp.Sum(row, output_y),
    )

    model.solve()

    assert np.allclose(output_y.toDense(), [1.5, 10, 6, 6, -6])
    if fct is pwl_interval_formulation:
        # The rays row uses one segment, and only that row has ray slots.
        assert segment_selector.up.toDense()[-1, 1:].tolist() == [0, 0, 0]
        assert np.allclose(ray_selector.up.toDense()[:-1], 0)

    input_x.fx["rays"] = 4
    model.solve()

    assert math.isclose(output_y.toDense()[-1], 2)


def test_pwl_indexed_parameter_broadcast():
    for formulation in fcts_to_test:
        m = gp.Container()
        product = gp.Set(m, "product", records=["basic", "premium"])
        breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2"])
        input_x = gp.Variable(m, "input_x", domain=product)
        x_points = gp.Parameter(
            m,
            "x_points",
            domain=breakpoint,
            records=[("p0", 0), ("p1", 2), ("p2", 4)],
        )
        y_points = gp.Parameter(
            m,
            "y_points",
            domain=breakpoint,
            records=[("p0", 1), ("p1", 5), ("p2", 9)],
        )
        input_level = gp.Parameter(
            m,
            "input_level",
            domain=product,
            records=[("basic", 1), ("premium", 3)],
        )

        output_y, equations = formulation(input_x, x_points, y_points)
        input_x.fx[product] = input_level[product]
        model = gp.Model(
            m,
            equations=equations,
            problem="mip",
            sense="min",
            objective=gp.Sum(product, output_y),
        )

        model.solve()

        assert np.allclose(output_y.toDense(), [3, 7])


@pytest.mark.parametrize(
    "fct",
    [
        pwl_interval_formulation,
        pwl_dlog_formulation,
        pwl_convexity_formulation,
    ],
    ids=["interval", "dlog", "convexity"],
)
def test_pwl_partial_parameter_broadcast(fct):
    m = gp.Container()
    scenario = gp.Set(m, "scenario", records=["low", "high"])
    product = gp.Set(m, "product", records=["bounded", "rays"])
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3"])
    input_x = gp.Variable(m, "input_x", domain=[scenario, product])
    x_points = gp.Parameter(
        m,
        "x_points",
        domain=[product, breakpoint],
        records=np.array([[0, 2, 2, 4], [-math.inf, 0, 2, math.inf]]),
    )
    y_points = gp.Parameter(
        m,
        "y_points",
        domain=[product, breakpoint],
        records=np.array([[0, 4, 6, 10], [3, 0, 4, -1]]),
    )
    input_level = gp.Parameter(
        m,
        "input_level",
        domain=[scenario, product],
        records=np.array([[1, -2], [3, 4]]),
    )

    output_y, equations = fct(
        input_x,
        x_points,
        y_points,
        allow_multivalued=True,
    )
    input_x.fx[scenario, product] = input_level[scenario, product]
    model = gp.Model(
        m,
        equations=equations,
        problem="mip",
        sense="max",
        objective=gp.Sum((scenario, product), output_y),
    )

    model.solve()

    assert output_y.domain == input_x.domain
    assert output_y.lo.toDense()[:, 0].tolist() == [0, 0]
    assert output_y.up.toDense()[:, 0].tolist() == [10, 10]
    assert np.allclose(output_y.toDense(), [[2, -6], [8, 2]])


@pytest.mark.parametrize(
    "fct",
    [
        pwl_interval_formulation,
        pwl_dlog_formulation,
        pwl_convexity_formulation,
    ],
    ids=["interval", "dlog", "convexity"],
)
def test_pwl_partial_parameter_domain_validation(fct):
    m = gp.Container()
    i = gp.Set(m, "i", records=["i0", "i1"])
    j = gp.Set(m, "j", records=["j0", "j1"])
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1"])
    input_x = gp.Variable(m, "input_x", domain=[i, j])
    x_points = gp.Parameter(m, "x_points", domain=[j, i, breakpoint])
    y_points = gp.Parameter(m, "y_points", domain=[j, i, breakpoint])

    symbols_before = list(m.data)
    with pytest.raises(ValidationError, match="ordered subset"):
        fct(input_x, x_points, y_points)
    assert list(m.data) == symbols_before


def test_pwl_scalar_convexity_normalized_solve():
    for using in ("binary", "sos2"):
        m = gp.Container()
        input_x = gp.Variable(m, "input_x")
        output_y, equations = pwl_convexity_formulation(
            input_x,
            [0, 2, 4],
            [0, 4, 6],
            using=using,
            bound_left=False,
            bound_right=False,
        )
        input_x.fx = -1
        model = gp.Model(
            m,
            equations=equations,
            problem="mip",
            sense="min",
            objective=output_y,
        )

        model.solve()

        assert math.isclose(output_y.toDense(), -2)

        input_x.fx = 6
        model.solve()

        assert math.isclose(output_y.toDense(), 8)


def test_pwl_indexed_convexity_normalized_solve():
    for using in ("binary", "sos2"):
        m = gp.Container()
        row = gp.Set(m, "row", records=["jump", "hole"])
        input_x = gp.Variable(m, "input_x", domain=row)
        output_y, equations = pwl_convexity_formulation(
            input_x,
            [0, 2, 2, None, 4, 6],
            [0, 4, 6, None, 8, 12],
            using=using,
            allow_multivalued=True,
        )
        input_x.fx["jump"] = 2
        input_x.fx["hole"] = 5
        model = gp.Model(
            m,
            equations=equations,
            problem="mip",
            sense="max",
            objective=gp.Sum(row, output_y),
        )

        model.solve()

        assert np.allclose(output_y.toDense(), [6, 10])


def test_pwl_heterogeneous_convexity_solve():
    for using in ("binary", "sos2"):
        m = gp.Container()
        row = gp.Set(
            m,
            "row",
            records=["long", "short", "hole", "jump", "rays"],
        )
        breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3", "p4"])
        input_x = gp.Variable(m, "input_x", domain=row)
        na = gp.SpecialValues.NA
        x_points = gp.Parameter(
            m,
            "x_points",
            domain=[row, breakpoint],
            records=np.array(
                [
                    [0, 1, 2, 3, 4],
                    [0, 2, 4, na, na],
                    [0, 2, na, 4, 6],
                    [0, 2, 2, 4, na],
                    [-math.inf, 0, 2, math.inf, na],
                ]
            ),
        )
        y_points = gp.Parameter(
            m,
            "y_points",
            domain=[row, breakpoint],
            records=np.array(
                [
                    [0, 1, 2, 3, 4],
                    [0, 4, 8, na, na],
                    [0, 4, na, 8, 12],
                    [0, 4, 6, 10, na],
                    [3, 0, 4, -1, na],
                ]
            ),
        )
        interval_y, interval_equations = pwl_interval_formulation(
            input_x,
            x_points,
            y_points,
            allow_multivalued=True,
        )
        variables_before = set(m.getVariables())

        output_y, equations = pwl_convexity_formulation(
            input_x,
            x_points,
            y_points,
            using=using,
            allow_multivalued=True,
        )

        new_variables = set(m.getVariables()) - variables_before
        lambda_var = next(
            variable
            for variable in new_variables
            if variable.type == ("free" if using == "binary" else "sos2")
            and variable.dimension == 2
            and len(variable.domain[-1]) == 5
        )
        ray_selectors = [
            variable
            for variable in new_variables
            if variable.type == "binary"
            and variable.dimension == 2
            and len(variable.domain[-1]) == 2
        ]
        input_x.fx[row] = gp.Parameter(
            m,
            "input_level",
            domain=row,
            records=[
                ("long", 1.5),
                ("short", 3),
                ("hole", 5),
                ("jump", 2),
                ("rays", -2),
            ],
        )
        model = gp.Model(
            m,
            equations=[*equations, *interval_equations],
            problem="mip",
            sense="max",
            objective=gp.Sum(row, output_y + interval_y),
        )

        model.solve()

        assert np.allclose(output_y.toDense(), [1.5, 6, 10, 6, -6])
        assert np.allclose(output_y.toDense(), interval_y.toDense())
        assert lambda_var.up.toDense()[1, 3:].tolist() == [0, 0]
        assert lambda_var.up.toDense()[-1, 2:].tolist() == [0, 0, 0]
        assert any(
            np.allclose(variable.up.toDense()[:-1], 0) for variable in ray_selectors
        )

        input_x.fx["rays"] = 4
        model.solve()

        assert math.isclose(output_y.toDense()[-1], 2)
        assert math.isclose(interval_y.toDense()[-1], 2)


def test_pwl_heterogeneous_convexity_overlap_validation():
    m = gp.Container()
    row = gp.Set(m, "row", records=["valid", "overlap"])
    breakpoint = gp.Set(m, "breakpoint", records=["p0", "p1", "p2", "p3"])
    input_x = gp.Variable(m, "input_x", domain=row)
    x_points = gp.Parameter(
        m,
        "x_points",
        domain=[row, breakpoint],
        records=np.array([[0, 2, 4, 6], [0, 4, 2, 6]]),
    )
    y_points = gp.Parameter(
        m,
        "y_points",
        domain=[row, breakpoint],
        records=np.array([[0, 4, 8, 12], [0, 4, 5, 9]]),
    )
    symbols_before = list(m.data)

    with pytest.raises(
        ValidationError,
        match=r"for row='overlap'.*allow_multivalued=True",
    ):
        pwl_convexity_formulation(input_x, x_points, y_points)

    assert list(m.data) == symbols_before


@pytest.mark.parametrize(
    ("method", "using"),
    [
        ("interval", None),
        ("convexity", None),
        ("convexity", "binary"),
        ("convexity", "sos2"),
        ("dlog", None),
    ],
)
def test_pwlinear_curve_structures(method, using):
    m = gp.Container()
    row_records = ["connected", "hole", "jump", "left_ray", "right_ray"]
    if method != "convexity":
        row_records.append("overlap")

    row = gp.Set(m, records=row_records)
    input_x = gp.Variable(m, domain=row)
    curves = {
        "connected": PWLCurve([(0, 0), (2, 4), (4, 8)]),
        "hole": PWLCurve([(0, 0), (2, 4), None, (4, 8), (6, 12)]),
        "jump": PWLCurve([(0, 0), (2, 4), (2, 6), (4, 10)]),
        "left_ray": PWLCurve([(0, 0)], left_gradient=3),
        "right_ray": PWLCurve([(0, 0), (2, 4)], right_gradient=-1),
    }
    input_levels = gp.Parameter(
        m,
        domain=row,
        records=[
            ("connected", 1),
            ("hole", 5),
            ("jump", 2),
            ("left_ray", -2),
            ("right_ray", 4),
        ],
    )
    expected = [2, 10, 6, -6, 2]
    if method != "convexity":
        curves["overlap"] = PWLCurve([(0, 0), (4, 4), (2, 5), (6, 9)])
        input_levels["overlap"] = 3
        expected.append(6)

    output_y, equations = pwlinear(
        input_x,
        curves,
        method=method,
        using=using,
        allow_multivalued=True,
    )
    input_x.fx[row] = input_levels[row]
    model = gp.Model(
        m,
        equations=equations,
        problem="mip",
        sense="max",
        objective=gp.Sum(row, output_y),
    )

    model.solve()

    assert output_y.domain == input_x.domain
    assert np.allclose(output_y.toDense(), expected)
    if method == "convexity":
        variable_types = {variable.type for variable in m.getVariables()}
        expected_type = "sos2" if using == "sos2" else "binary"
        assert expected_type in variable_types

    input_x.fx["hole"] = 3
    result = model.solve()

    assert result["Model Status"].item() == "IntegerInfeasible"


@pytest.mark.parametrize("method", ["interval", "convexity", "dlog"])
def test_pwlinear_partial_curve_mapping(method):
    m = gp.Container()
    scenario = gp.Set(m, "scenario", records=["low", "high"])
    product = gp.Set(m, "product", records=["bounded", "rays"])
    input_x = gp.Variable(m, "input_x", domain=[scenario, product])
    curves = {
        "bounded": PWLCurve([(0, 0), (2, 4), (2, 6), (4, 10)]),
        "rays": PWLCurve(
            [(0, 0), (2, 4)],
            left_gradient=3,
            right_gradient=-1,
        ),
    }
    input_level = gp.Parameter(
        m,
        "input_level",
        domain=[scenario, product],
        records=np.array([[1, -2], [3, 4]]),
    )

    output_y, equations = pwlinear(
        input_x,
        curves,
        curve_domain=[product],
        method=method,
        allow_multivalued=True,
    )
    input_x.fx[scenario, product] = input_level[scenario, product]
    model = gp.Model(
        m,
        equations=equations,
        problem="mip",
        sense="max",
        objective=gp.Sum((scenario, product), output_y),
    )

    model.solve()

    assert output_y.domain == input_x.domain
    assert np.allclose(output_y.toDense(), [[2, -6], [8, 2]])


def test_pwlinear_validation():
    m = gp.Container()
    input_x = gp.Variable(m)
    curve = PWLCurve([(0, 0), (1, 1)])
    symbols_before = list(m.data)

    with pytest.raises(ValidationError, match="method argument"):
        pwlinear(input_x, curve, method="convex")
    with pytest.raises(ValidationError, match="using argument"):
        pwlinear(input_x, curve, method="convexity", using="logarithmic")
    with pytest.raises(ValidationError, match="allow_multivalued"):
        pwlinear(input_x, curve, allow_multivalued="yes")
    with pytest.raises(ValidationError, match="curve must be a PWLCurve"):
        pwlinear(input_x, [(0, 0), (1, 1)])

    overlapping_curve = PWLCurve([(0, 0), (4, 4), (2, 5), (6, 9)])
    with pytest.raises(ValidationError, match="cannot represent overlapping"):
        pwlinear(input_x, overlapping_curve, method="convexity")

    two_rays_one_point = PWLCurve(
        [(0, 0)],
        left_gradient=1,
        right_gradient=2,
    )
    with pytest.raises(
        ValidationError,
        match="left and right rays sharing one finite point",
    ):
        pwlinear(input_x, two_rays_one_point, method="convexity")

    assert list(m.data) == symbols_before

    empty_container = gp.Container()
    empty_domain = gp.Set(empty_container)
    empty_x = gp.Variable(empty_container, domain=empty_domain)
    empty_symbols_before = list(empty_container.data)
    with pytest.raises(ValidationError, match="domain cannot be empty"):
        pwlinear(empty_x, {})
    assert list(empty_container.data) == empty_symbols_before

    wildcard_container = gp.Container()
    wildcard_x = gp.Variable(wildcard_container, domain=[gp.UNIVERSE])
    wildcard_symbols_before = list(wildcard_container.data)
    with pytest.raises(ValidationError, match="explicit Sets or Aliases"):
        pwlinear(wildcard_x, curve)
    assert list(wildcard_container.data) == wildcard_symbols_before

    with pytest.raises(ValidationError, match="at least 2 points"):
        PWLCurve([(0, 0)])
    with pytest.raises(ValidationError, match=r"must be an \(x, y\) pair"):
        PWLCurve([(0, 0, 1), (1, 1)])
    with pytest.raises(ValidationError, match="finite real number"):
        PWLCurve([(0, 0), (1, float("inf"))])
    with pytest.raises(ValidationError, match="finite real number"):
        PWLCurve([(0, 0), (1, 1)], left_gradient=True)
    with pytest.raises(ValidationError, match="two consecutive None"):
        PWLCurve([(0, 0), None, None, (1, 1)])

    product = gp.Set(m, records=["basic", "premium"])
    indexed_x = gp.Variable(m, domain=product)
    symbols_before = list(m.data)
    with pytest.raises(ValidationError, match="missing keys: 'premium'"):
        pwlinear(indexed_x, {"basic": curve})
    with pytest.raises(ValidationError, match="extra keys: 'other'"):
        pwlinear(
            indexed_x,
            {"basic": curve, "premium": curve, "other": curve},
        )
    with pytest.raises(ValidationError, match="key 'premium' must be a PWLCurve"):
        pwlinear(
            indexed_x,
            {"basic": curve, "premium": [(0, 0), (1, 1)]},
        )
    with pytest.raises(ValidationError, match="only be used with a curve dictionary"):
        pwlinear(indexed_x, curve, curve_domain=[product])

    assert list(m.data) == symbols_before

    scenario = gp.Set(m, records=["low", "high"])
    matrix_x = gp.Variable(m, domain=[scenario, product])
    partial_curves = {"basic": curve, "premium": curve}
    symbols_before = list(m.data)
    with pytest.raises(ValidationError, match="ordered subset"):
        pwlinear(
            matrix_x,
            partial_curves,
            curve_domain=[product, scenario],
        )
    with pytest.raises(ValidationError, match="missing keys: 'premium'"):
        pwlinear(
            matrix_x,
            {"basic": curve},
            curve_domain=[product],
        )
    assert list(m.data) == symbols_before

    for method in ("interval", "dlog"):
        warning_container = gp.Container()
        warning_x = gp.Variable(warning_container)
        with pytest.warns(UserWarning, match="using argument is ignored"):
            output_y, equations = pwlinear(
                warning_x,
                curve,
                method=method,
                using="sos2",
            )
        assert output_y.domain == warning_x.domain
        assert equations
