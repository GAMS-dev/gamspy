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


def _enforce_sos2_with_binary(lambda_var: gp.Variable) -> list[gp.Equation]:
    """
    Enforces SOS2 constraints using binary variables.

    Based on paper:
    `Modeling disjunctive constraints with a logarithmic number of binary variables and constraints
    <https://www.researchgate.net/publication/225976267_Modeling_Disjunctive_Constraints_with_a_Logarithmic_Number_of_Binary_Variables_and_Constraints>`_
    """
    equations: list[gp.Equation] = []
    m = lambda_var.container
    count_x = len(lambda_var.domain[-1])
    # edge case
    if count_x == 2:
        # if there are only 2 elements, it is already sos2
        return equations

    J = lambda_var.domain[-1]
    previous_domains = lambda_var.domain[:-1]

    l_len = math.ceil(math.log2(count_x - 1))
    I, L = gp.math._generate_dims(
        m,
        [
            count_x - 1,
            l_len,
        ],
    )

    J, I, L = gp.formulations.nn.utils._next_domains(
        [J, I, L], previous_domains
    )
    bin_var = m.addVariable(domain=[*previous_domains, L], type="binary")
    gray_code = _generate_gray_code(count_x - 1, l_len)

    B = m.addParameter(domain=[I, L], records=gray_code)

    JI = m.addSet(domain=[J, I])
    JI[J, I].where[(gp.Ord(J) == gp.Ord(I)) | (gp.Ord(J) - 1 == gp.Ord(I))] = 1

    use_set_1 = m.addSet(domain=[L, J])
    use_set_1[L, J].where[gp.Smin(JI[J, I], B[I, L]) == 1] = 1

    use_set_2 = m.addSet(domain=[L, J])
    use_set_2[L, J].where[gp.Smax(JI[J, I], B[I, L]) == 0] = 1

    sos2_eq_1 = m.addEquation(domain=[*previous_domains, L])
    sos2_eq_1[[*previous_domains, L]] = (
        gp.Sum(use_set_1[L, J], lambda_var[[*previous_domains, J]])
        <= bin_var[*previous_domains, L]
    )
    equations.append(sos2_eq_1)

    sos2_eq_2 = m.addEquation(domain=[*previous_domains, L])
    sos2_eq_2[[*previous_domains, L]] = (
        gp.Sum(use_set_2[L, J], lambda_var[[*previous_domains, J]])
        <= 1 - bin_var[*previous_domains, L]
    )
    equations.append(sos2_eq_2)

    return equations


def _enforce_discontinuity(
    lambda_var: gp.Variable,
    combined_indices: typing.Sequence[int],
) -> list[gp.Equation]:
    equations: list[gp.Equation] = []

    len_x_points = len(lambda_var.domain[-1])
    previous_domains = lambda_var.domain[:-1]

    m = lambda_var.container
    J, J2, SB = gp.math._generate_dims(
        m, [len_x_points, len_x_points, len(combined_indices)]
    )

    J, J2, SB = gp.formulations.nn.utils._next_domains(
        [J, J2, SB], previous_domains
    )

    di_param = [
        (str(i), str(j), str(j + 1)) for i, j in enumerate(combined_indices)
    ]

    select_set = m.addSet(domain=[SB, J, J2], records=di_param)
    select_var = m.addVariable(domain=[*previous_domains, SB], type="binary")

    select_equation = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation[[*previous_domains, select_set[SB, J, J2]]] = (
        lambda_var[[*previous_domains, J]] <= select_var[*previous_domains, SB]
    )
    equations.append(select_equation)

    select_equation_2 = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation_2[[*previous_domains, select_set[SB, J, J2]]] = (
        lambda_var[[*previous_domains, J2]]
        <= 1 - select_var[*previous_domains, SB]
    )
    equations.append(select_equation_2)

    return equations


def _check_points(
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
) -> tuple[list[int | float], list[int | float], list[int], list[int]]:
    return_x = []
    return_y = []
    discontinuous_indices = []
    none_indices = []

    if not isinstance(x_points, typing.Sequence):
        raise ValidationError("x_points are expected to be a sequence")

    if not isinstance(y_points, typing.Sequence):
        raise ValidationError("y_points are expected to be a sequence")

    if len(x_points) < 2:
        raise ValidationError(
            "piecewise linear functions require at least 2 points"
        )

    if len(y_points) != len(x_points):
        raise ValidationError("x_points and y_points have different lenghts")

    if x_points[0] is None or x_points[-1] is None:
        raise ValidationError("x_points cannot start or end with a None value")

    for x_p, y_p in zip(x_points, y_points):
        if (x_p is None and y_p is not None) or (
            x_p is not None and y_p is None
        ):
            raise ValidationError(
                "Both x and y must either be None or neither of them should be None"
            )

        if not isinstance(x_p, (float, int)) and x_p is not None:
            raise ValidationError("x_points contains non-numerical items")

        if not isinstance(y_p, (float, int)) and y_p is not None:
            raise ValidationError("y_points contains non-numerical items")

    for i in range(len(x_points) - 1):
        if x_points[i] is None and x_points[i + 1] is None:
            raise ValidationError(
                "x_points cannot contain two consecutive None values"
            )

        if x_points[i] is None and x_points[i - 1] >= x_points[i + 1]:
            raise ValidationError(
                "A value following a None must be strictly greater than the value preceding the None"
            )

        if (
            (x_points[i] is not None)
            and (x_points[i + 1] is not None)
            and (x_points[i + 1] < x_points[i])
        ):
            raise ValidationError(
                "x_points should be in an non-decreasing order"
            )

        if x_points[i] is not None:
            return_x.append(x_points[i])
            return_y.append(y_points[i])

        if x_points[i] == x_points[i + 1]:
            discontinuous_indices.append(len(return_x) - 1)
        elif x_points[i] is None:
            none_indices.append(len(return_x) - 1)

    return_x.append(x_points[-1])
    return_y.append(y_points[-1])
    return return_x, return_y, discontinuous_indices, none_indices


def piecewise_linear_function(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
    using: typing.Literal["binary", "sos2"] = "binary",
    bound_domain: bool = True,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    This function implements a piecewise linear function. Given an input
    (independent) variable `input_x`, along with the defining `x_points` and
    corresponding `y_points` of the piecewise function, it constructs the
    dependent variable `y` and formulates the equations necessary to define the
    function.

    Internally, the function uses binary variables by default. If preferred,
    you can switch to SOS2 (Special Ordered Set Type 2) by setting the `using`
    parameter to "sos2". `bound_domain` cannot be set to False while using
    binary variable implementation.

    The implementation handles discontinuities in the function. To represent a
    discontinuity at a specific point `x_i`, include `x_i` twice in the `x_points`
    array with corresponding values in `y_points`. For example, if `x_points` =
    [1, 3, 3, 5] and `y_points` = [10, 30, 50, 70], the function allows y to take
    either 30 or 50 when x = 3. Note that discontinuities always introduce
    additional binary variables, regardless of the value of the using argument.

    It is possible to disallow a specific range by including `None` in both
    `x_points` and the corresponding `y_points`. For example, with
    `x_points` = `[1, 3, None, 5, 7]` and `y_points` = `[10, 35, None, -20, 40]`,
    the range between 3 and 5 is disallowed for `input_x`.

    However, `x_points` cannot start or end with a `None` value, and a `None`
    value cannot be followed by another `None`. Additionally, if `x_i` is `None`,
    then `y_i` must also be `None`. Similar to the discontinuities, disallowed
    ranges always introduce additional binary variables, regardless of the value
    of the using argument.

    The input variable `input_x` is restricted to the range defined by
    `x_points` unless `bound_domain` is set to False. `bound_domain` can be set
    to False only if using is "sos2". When `input_x` is not bound, you can assume
    as if the first and the last line segments are extended.

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
    using: str = "binary"
        What type of variable is used during implementing piecewise function
    bound_domain: bool = True
        If input_x should be limited to interval defined by min(x_points), max(x_points)

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

    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if bound_domain is False and using == "binary":
        raise ValidationError(
            "bound_domain can only be false when using is sos2"
        )

    x_points, y_points, discontinuous_indices, none_indices = _check_points(
        x_points, y_points
    )
    combined_indices = list({*discontinuous_indices, *none_indices})

    m = input_x.container
    input_domain = input_x.domain
    out_y = m.addVariable(domain=input_domain)
    equations = []

    J = gp.math._generate_dims(m, [len(x_points)])[0]
    J = gp.formulations.nn.utils._next_domains([J], input_domain)[0]

    x_par = m.addParameter(domain=[J], records=np.array(x_points))
    y_par = m.addParameter(domain=[J], records=np.array(y_points))

    lambda_var = m.addVariable(
        domain=[*input_domain, J], type="free" if using == "binary" else "sos2"
    )

    lambda_var.lo[...] = 0
    lambda_var.up[...] = 1
    if not bound_domain:
        # lower bounds
        lambda_var.lo[*input_domain, J].where[gp.Ord(J) == 2] = float("-inf")
        lambda_var.lo[*input_domain, J].where[gp.Ord(J) == gp.Card(J) - 1] = (
            float("-inf")
        )

        # upper bound
        lambda_var.up[*input_domain, J].where[gp.Ord(J) == 1] = float("inf")
        lambda_var.up[*input_domain, J].where[gp.Ord(J) == gp.Card(J)] = float(
            "inf"
        )
    else:
        min_y = min(y_points)
        max_y = max(y_points)
        out_y.lo[...] = min_y
        out_y.up[...] = max_y

    lambda_sum = m.addEquation(domain=input_x.domain)
    lambda_sum[...] = gp.Sum(J, lambda_var) == 1
    equations.append(lambda_sum)

    set_x = m.addEquation(domain=input_x.domain)
    set_x[...] = input_x == gp.Sum(J, x_par * lambda_var)
    equations.append(set_x)

    set_y = m.addEquation(domain=input_x.domain)
    set_y[...] = out_y == gp.Sum(J, y_par * lambda_var)
    equations.append(set_y)

    if using == "binary":
        extra_eqs = _enforce_sos2_with_binary(lambda_var)
        equations.extend(extra_eqs)

    if len(combined_indices) > 0:
        extra_eqs = _enforce_discontinuity(lambda_var, combined_indices)
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
    bound_domain: bool = True,
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
    return piecewise_linear_function(
        input_x, x_points, y_points, using="sos2", bound_domain=bound_domain
    )
