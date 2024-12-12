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


def _enforce_sos2_with_binary(lambda_var: gp.Variable):
    """
    Enforces SOS2 constraints using binary variables.

    Based on paper:
    `Modeling disjunctive constraints with a logarithmic number of binary variables and constraints
    <https://www.academia.edu/download/43527291/Modeling_Disjunctive_Constraints_with_a_20160308-26796-1g6hb4g.pdf>`_
    """
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

    return equations


def _check_points(
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
) -> None:
    if not isinstance(x_points, typing.Sequence):
        raise ValidationError("x_points are expected to be a sequence")

    if not isinstance(y_points, typing.Sequence):
        raise ValidationError("y_points are expected to be a sequence")

    if len(x_points) <= 2:
        raise ValidationError(
            "piecewise linear functions require at least 2 points"
        )

    if len(y_points) != len(x_points):
        raise ValidationError("x_points and y_points have different lenghts")

    for li, name in [(x_points, "x_points"), (y_points, "y_points")]:
        for item in li:
            if not isinstance(item, (float, int)):
                raise ValidationError(f"{name} contains non-numerical items")

    for i in range(len(x_points) - 1):
        if x_points[i + 1] < x_points[i]:
            raise ValidationError(
                "x_points should be in an non-decreasing order"
            )


def piecewise_linear_function(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
    using: typing.Literal["binary", "sos2"] = "sos2",
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    This function implements a piecewise linear function. Given an input
    (independent) variable `input_x`, along with the defining `x_points` and
    corresponding `y_points` of the piecewise function, it constructs the
    dependent variable `y` and formulates the equations necessary to define the
    function.

    The implementation supports discontinuities. If the function is
    discontinuous at a specific point `x_i`, you can specify `x_i` twice in
    `x_points` with distinct y_points values. For instance, with `x_points`
    = `[1, 3, 3, 5]` and `y_points` = `[10, 30, 50, 70]`, the function allows
    `y` to take either 30 or 50 at `x=3`.

    The input variable `input_x` is restricted to the range defined by
    `x_points`.

    Internally, the function uses SOS2 (Special Ordered Set Type 2) variables
    by default. If preferred, you can switch to binary variables by setting the
    `using` parameter to "binary".

    Returns the dependent variable `y` and the equations required to model the
    piecewise linear relationship.

    Parameters
    ----------
    x : gp.Variable
        Independent variable of the piecewise linear function
    x_points: typing.Sequence[int | float]
        Break points of the piecewise linear function in the x-axis
    y_points: typing.Sequence[int| float]
        Break points of the piecewise linear function in the y-axis
    using: str = "sos2"

    Returns
    -------
    tuple[gp.Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.formulations import piecewise_linear_function
    >>> m = Container()
    >>> x = Variable(m, "x")
    >>> y, eqs = piecewise_linear_function(x, [-1, 4, 10, 10, 20], [-2, 8, 15, 17, 37])

    """
    if using not in {"binary", "sos2"}:
        raise ValidationError(
            "Invalid value for the using argument."
            "Possible values are 'binary' and 'sos2'"
        )

    _check_points(x_points, y_points)

    m = input_x.container
    out_y = m.addVariable()
    equations = []

    J = gp.math._generate_dims(m, [len(x_points)])[0]
    x_par = m.addParameter(domain=[J], records=np.array(x_points))
    y_par = m.addParameter(domain=[J], records=np.array(y_points))

    min_y = min(y_points)
    max_y = max(y_points)
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
        extra_eqs = _enforce_sos2_with_binary(lambda_var)
        equations.extend(extra_eqs)

    return out_y, equations


def piecewise_linear_function_with_binary(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    Calls the piecewise_linear_function setting `using` keyword argument
    to `binary`

    Parameters
    ----------
    x : gp.Variable
        Independent variable of the piecewise linear function
    x_points: typing.Sequence[int | float]
        Break points of the piecewise linear function in the x-axis
    y_points: typing.Sequence[int| float]
        Break points of the piecewise linear function in the y-axis

    """
    return piecewise_linear_function(
        input_x, x_points, y_points, using="binary"
    )


def piecewise_linear_function_with_sos2(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    Calls the piecewise_linear_function setting `using` keyword argument
    to `sos2`.

    Parameters
    ----------
    x : gp.Variable
        Independent variable of the piecewise linear function
    x_points: typing.Sequence[int | float]
        Break points of the piecewise linear function in the x-axis
    y_points: typing.Sequence[int| float]
        Break points of the piecewise linear function in the y-axis

    """
    return piecewise_linear_function(input_x, x_points, y_points, using="sos2")
