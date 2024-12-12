from __future__ import annotations

import pytest

import gamspy as gp
from gamspy.exceptions import ValidationError
from gamspy.formulations import (
    piecewise_linear_function,
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
