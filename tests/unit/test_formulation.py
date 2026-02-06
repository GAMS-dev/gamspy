from __future__ import annotations

import pytest

import gamspy as gp
import gamspy.formulations.piecewise as piecewise
from gamspy.exceptions import ValidationError
from gamspy.formulations import (
    pwl_convexity_formulation,
    pwl_interval_formulation,
)

pytestmark = pytest.mark.unit

fcts_to_test = [
    pwl_convexity_formulation,
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


def test_pwl_with_domain(data):
    x2 = data["x2"]
    x_points = data["x_points"]
    y_points = data["y_points"]
    y, _eqs = pwl_convexity_formulation(x2, x_points, y_points, using="binary")
    y2, _eqs2 = pwl_convexity_formulation(x2, x_points, y_points, using="sos2")
    y3, _eqs3 = pwl_interval_formulation(x2, x_points, y_points)

    assert len(y.domain) == len(x2.domain)
    assert len(y2.domain) == len(x2.domain)
    assert len(y3.domain) == len(x2.domain)


def test_pwl_with_none(data):
    x = data["x"]
    x_points = [1, None, 2, 3]
    y_points = [10, None, 20, 45]
    _y, _eqs = pwl_convexity_formulation(x, x_points, y_points)
    _y2, _eqs2 = pwl_interval_formulation(x, x_points, y_points)


def test_pwl_finished_start_with_disc(data):
    x = data["x"]
    x_points = [1, 1, None, 2, 3, 3]
    y_points = [0, 10, None, 20, 45, 0]
    _y, _eqs = pwl_convexity_formulation(
        x,
        x_points,
        y_points,
        bound_left=False,
        bound_right=False,
    )
    _y2, _eqs2 = pwl_interval_formulation(
        x,
        x_points,
        y_points,
        bound_left=False,
        bound_right=False,
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
        fct(x, x_points, y_points, "hello")

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

    with pytest.raises(ValidationError):
        fct(x, [3, 2, 1], [10, 20, 30])

    with pytest.raises(ValidationError):
        fct(x, [3, 1, 2], [10, 20, 30])

    with pytest.raises(ValidationError):
        fct(x, [1, 3, 2], [10, 20, 30])

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
