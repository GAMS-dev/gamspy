from __future__ import annotations

import math
import typing

import numpy as np

import gamspy as gp
from gamspy.exceptions import ValidationError

number = typing.Union[int, float]


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


def _check_points(
    x_to_fx: dict[number, number],
) -> tuple[list[number], list[number]]:
    if not isinstance(x_to_fx, dict):
        raise ValidationError("Function mapping must be a dictionary")

    x_vals = []
    y_vals = []

    old_k = None
    for k in x_to_fx:
        if not isinstance(k, (float, int)):
            raise ValidationError("Keys need to be float or integer")

        v = x_to_fx[k]
        if not isinstance(v, (float, int)):
            raise ValidationError("Values need to be float or integer")

        if old_k is None:
            old_k = k
        elif k <= old_k:
            raise ValidationError("Keys need to be sorted")

        x_vals.append(k)
        y_vals.append(v)

    return x_vals, y_vals


def enforce_sos2_with_binary(lambda_var: gp.Variable):
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
    points: dict[number, number],
    using: typing.Literal["binary", "sos2"] = "sos2",
) -> tuple[gp.Variable, list[gp.Equation]]:
    if using not in {"binary", "sos2"}:
        raise ValidationError(
            "Invalid value for the using argument."
            "Possible values are 'binary' and 'sos2'"
        )

    x_vals, y_vals = _check_points(points)

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
        _, extra_eqs = enforce_sos2_with_binary(lambda_var)
        equations.extend(extra_eqs)

    return out_y, equations


def piecewise_linear_function_with_binary(
    input_x: gp.Variable,
    points: dict[number, number],
) -> tuple[gp.Variable, list[gp.Equation]]:
    return piecewise_linear_function(input_x, points, using="binary")


def piecewise_linear_function_with_sos2(
    input_x: gp.Variable,
    points: dict[number, number],
) -> tuple[gp.Variable, list[gp.Equation]]:
    return piecewise_linear_function(input_x, points, using="sos2")
