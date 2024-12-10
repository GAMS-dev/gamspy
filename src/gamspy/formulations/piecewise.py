from __future__ import annotations

import math
import typing

import numpy as np

import gamspy as gp
from gamspy.exceptions import ValidationError

number = typing.Union[int, float]
linear_expression = typing.Union["gp.Expression", "gp.Variable", number]


def _generate_gray_code(n: int, n_bits: int) -> np.ndarray:
    """
    Returns an n x n_bits NumPy array containing gray codes.
    The bit difference between two consecutive rows is exactly
    1 bits. Required for the log piecewise linear formulation.
    """
    a = np.arange(n)
    b = a >> 1
    numbers = a ^ b
    numbers_in_bit_array = (
        (numbers[:, None] & (1 << np.arange(n_bits))) > 0
    ).astype(int)
    return numbers_in_bit_array


def _get_linear_coefficients(expr: linear_expression) -> tuple[float, float]:
    """
    Assuming the provided expression is in shape y = mx +n, it returns the
    coefficients m, n
    """
    # constant y = c
    if isinstance(expr, (int, float)):
        return 0, expr

    # y = x
    if isinstance(expr, gp.Variable):
        return 1, 0

    # TODO implement
    return 1, 2


def _check_points(
    intervals: dict[tuple[number, number], linear_expression],
) -> tuple[list[number], list[number]]:
    if not isinstance(intervals, dict):
        raise ValidationError("Function mapping must be a dictionary")

    last_b = None
    last_y = None
    x_vals = []
    y_vals = []

    for a, b in intervals:
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValidationError(
                "Intervals must be specified using integers or floats"
            )

        if a > b:
            raise ValidationError("Interval's start is greater than its end")

        if last_b is None:
            last_b = a

        # TODO maybe we will relax it
        if last_b != a:
            raise ValidationError("Intervals cannot have any gap")

        last_b = b
        expr = intervals[(a, b)]
        if not isinstance(expr, (int, float, gp.Expression, gp.Variable)):
            raise ValidationError("Expression was in an unrecognized format")

        m, n = _get_linear_coefficients(expr)

        y1 = m * a + n
        y2 = m * b + n

        # discontinuity
        if last_y != y1:
            x_vals.append(a)
            y_vals.append(y1)

        x_vals.append(b)
        y_vals.append(y2)
        last_y = y2

    return x_vals, y_vals


def _enforce_sos2_with_binary(lambda_var: gp.Variable):
    equations = []
    m = lambda_var.container
    count_x = len(lambda_var.domain[-1])

    J = lambda_var.domain[-1]

    l_len = math.ceil(math.log2(count_x - 1))
    I, L = gp.math._generate_dims(
        m,
        [
            count_x - 1,
            l_len,
        ],
    )

    bin_var = m.addVariable(domain=[L], type="binary")
    gray_code = _generate_gray_code(count_x - 1, l_len)

    B = m.addParameter(domain=[I, L], records=gray_code)

    JI = m.addSet(domain=[J, I])
    JI[J, I].where[(gp.Ord(J) == gp.Ord(I)) | (gp.Ord(J) - 1 == gp.Ord(I))] = 1

    use_set_1 = m.addSet(domain=[L, J])
    use_set_1[L, J].where[gp.Smin(JI[J, I], B[I, L]) == 1] = 1

    use_set_2 = m.addSet(domain=[L, J])
    use_set_2[L, J].where[gp.Smax(JI[J, I], B[I, L]) == 0] = 1

    sos2_eq_1 = m.addEquation(domain=[L])
    sos2_eq_1[L] = gp.Sum(use_set_1[L, J], lambda_var[J]) <= bin_var[L]
    equations.append(sos2_eq_1)

    sos2_eq_2 = m.addEquation(domain=[L])
    sos2_eq_2[L] = gp.Sum(use_set_2[L, J], lambda_var[J]) <= 1 - bin_var[L]
    equations.append(sos2_eq_2)

    return lambda_var, equations


def piecewise_linear_function(
    input_x: gp.Variable,
    intervals: dict[tuple[number, number], linear_expression],
    using: typing.Literal["binary", "sos2"] = "sos2",
) -> tuple[gp.Variable, list[gp.Equation]]:
    if using not in {"binary", "sos2"}:
        raise ValidationError(
            "Invalid value for the using argument."
            "Possible values are 'binary' and 'sos2'"
        )

    x_vals, y_vals = _check_points(intervals)

    m = input_x.container
    out_y = m.addVariable()
    equations = []

    J = gp.math._generate_dims(m, [len(x_vals)])[0]
    x_par = m.addParameter(domain=[J], records=np.array(x_vals))
    y_par = m.addParameter(domain=[J], records=np.array(y_vals))

    min_y = min(y_vals)
    max_y = max(y_vals)
    out_y.lo[...] = min_y
    out_y.up[...] = max_y

    lambda_var = m.addVariable(
        domain=[J], type="free" if using == "binary" else "sos2"
    )
    lambda_var.lo[...] = 0
    lambda_var.up[...] = 1

    lambda_sum = m.addEquation()
    lambda_sum[...] = gp.Sum(J, lambda_var) == 1
    equations.append(lambda_sum)

    set_x = m.addEquation()
    set_x[...] = input_x == gp.Sum(J, x_par * lambda_var)
    equations.append(set_x)

    set_y = m.addEquation()
    set_y[...] = out_y == gp.Sum(J, y_par * lambda_var)
    equations.append(set_y)

    if using == "binary":
        _, extra_eqs = _enforce_sos2_with_binary(lambda_var)
        equations.extend(extra_eqs)

    return out_y, equations


def piecewise_linear_function_with_binary(
    input_x: gp.Variable,
    intervals: dict[tuple[number, number], linear_expression],
) -> tuple[gp.Variable, list[gp.Equation]]:
    return piecewise_linear_function(input_x, intervals, using="binary")


def piecewise_linear_function_with_sos2(
    input_x: gp.Variable,
    intervals: dict[tuple[number, number], linear_expression],
) -> tuple[gp.Variable, list[gp.Equation]]:
    return piecewise_linear_function(input_x, intervals, using="sos2")
