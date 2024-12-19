from __future__ import annotations

import pytest

import gamspy as gp
import gamspy.formulations.piecewise as piecewise
from gamspy.exceptions import ValidationError
from gamspy.formulations import (
    piecewise_linear_function,
    piecewise_linear_function_with_binary,
    piecewise_linear_function_with_sos2,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    m = gp.Container()
    x = gp.Variable(m, "x")
    x_points = [-10, 2.2, 5, 10]
    y_points = [10, 20, -2, -5]
    return {
        "m": m,
        "x": x,
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


def test_pwl_with_sos2(data):
    m = data["m"]
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]
    y, eqs = piecewise_linear_function_with_sos2(x, x_points, y_points)
    y2, eqs2 = piecewise_linear_function(x, x_points, y_points, using="sos2")
    y3, eqs2 = piecewise_linear_function_with_sos2(
        x, x_points, y_points, bound_domain=False
    )

    # there should be no binary variables
    var_count = get_var_count_by_type(m)
    assert "binary" not in var_count
    assert var_count["sos2"] == 3  # since we called it twice
    assert y.type == "free"
    assert y2.type == "free"
    assert y3.type == "free"
    assert len(eqs) == len(eqs2)


def test_pwl_with_binary(data):
    m = data["m"]
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]
    y, eqs = piecewise_linear_function_with_binary(x, x_points, y_points)
    y2, eqs2 = piecewise_linear_function(x, x_points, y_points, using="binary")

    # there should be no sos2 variables
    var_count = get_var_count_by_type(m)
    assert "sos2" not in var_count
    assert var_count["binary"] == 2  # since we called it twice
    assert y.type == "free"
    assert y2.type == "free"
    assert len(eqs) == len(eqs2)


def test_pwl_with_none(data):
    x = data["x"]
    x_points = [1, None, 2, 3]
    y_points = [10, None, 20, 45]
    y, eqs = piecewise_linear_function(x, x_points, y_points)


def test_pwl_validation(data):
    x = data["x"]
    x_points = data["x_points"]
    y_points = data["y_points"]

    # incorrect using value
    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        x_points,
        y_points,
        "hello",
    )

    # x not a variable
    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        10,
        x_points,
        y_points,
    )

    # incorrect x_points, y_points
    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        10,
        y_points,
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        x_points,
        10,
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [1],
        [10],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        x_points,
        [10],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [*x_points, "a"],
        [*y_points, 5],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [*x_points, 16],
        [*y_points, "a"],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [3, 2, 1],
        [10, 20, 30],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [3, 1, 2],
        [10, 20, 30],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [1, 3, 2],
        [10, 20, 30],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [1],
        [10],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [1, 2, 3],
        [10, 20, 40],
        using="binary",
        bound_domain=False,
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [None, 2, 3],
        [None, 20, 40],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [2, 3, None],
        [20, 40, None],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [None, 2, 3, None],
        [None, 20, 40, None],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [0, None, 2, 3],
        [0, 10, 20, 40],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [0, 1, 2, 3],
        [0, None, 20, 40],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [1, None, None, 2, 3],
        [10, None, None, 20, 40],
    )

    pytest.raises(
        ValidationError,
        piecewise_linear_function,
        x,
        [2, None, 2, 3],
        [10, None, 20, 40],
    )
