from __future__ import annotations

import math
import typing

import numpy as np

import gamspy as gp
from gamspy._symbols.implicits import (
    ImplicitVariable,
)
from gamspy.exceptions import ValidationError


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
    Enforces SOS2 constraints using binary variables. This function is not suitable
    for generic SOS2 implementation since it restricts the lambda_var values to be
    between 0 and 1. However, it is usually faster than using SOS2 variables.

    Based on paper:
    `Modeling disjunctive constraints with a logarithmic number of binary variables and constraints
    <https://link.springer.com/article/10.1007/s10107-009-0295-4>`_
    """
    equations: list[gp.Equation] = []
    m = lambda_var.container
    count_x = len(lambda_var.domain[-1])
    # edge case
    lambda_var.lo[...] = 0
    lambda_var.up[...] = 1
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
        <= bin_var[[*previous_domains, L]]
    )
    equations.append(sos2_eq_1)

    sos2_eq_2 = m.addEquation(domain=[*previous_domains, L])
    sos2_eq_2[[*previous_domains, L]] = (
        gp.Sum(use_set_2[L, J], lambda_var[[*previous_domains, J]])
        <= 1 - bin_var[[*previous_domains, L]]
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
        lambda_var[[*previous_domains, J]]
        <= select_var[[*previous_domains, SB]]
    )
    equations.append(select_equation)

    select_equation_2 = m.addEquation(domain=[*previous_domains, SB, J, J2])
    select_equation_2[[*previous_domains, select_set[SB, J, J2]]] = (
        lambda_var[[*previous_domains, J2]]
        <= 1 - select_var[[*previous_domains, SB]]
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


def _indicator(
    indicator_var: gp.Variable,
    indicator_val: typing.Literal[0, 1],
    expr: gp.Expression,
) -> list[gp.Equation]:
    # We will make this generic and public
    if not isinstance(indicator_var, (gp.Variable, ImplicitVariable)):
        raise ValidationError("indicator_var needs to be a variable")

    if indicator_var.type != "binary":
        raise ValidationError("indicator_var needs to be a binary variable")

    if indicator_val not in (0, 1):
        raise ValidationError("indicator_val needs to be 1 or 0")

    if not isinstance(expr, gp.Expression):
        raise ValidationError("expr needs to be an expression")

    if expr.data not in {"=l=", "=e=", "=g="}:
        raise ValidationError("expr needs to be inequality or equality")

    if len(expr.domain) != len(indicator_var.domain):
        raise ValidationError(
            "indicator_var and expr must have the same domain"
        )

    for i in range(len(expr.domain)):
        if expr.domain[i].name != indicator_var.domain[i].name:
            raise ValidationError(
                "indicator_var and expr must have the same domain"
            )

    if expr.data == "=e=":
        # sos1(bin_var, lhs - rhs) might be better
        eqs1 = _indicator(
            indicator_var,
            indicator_val,
            expr.left <= expr.right,  # type: ignore
        )
        eqs2 = _indicator(
            indicator_var,
            indicator_val,
            -expr.left <= -expr.right,  # type: ignore
        )
        return [*eqs1, *eqs2]

    if expr.data == "=g=":
        return _indicator(
            indicator_var,
            indicator_val,
            -expr.left <= -expr.right,  # type: ignore
        )

    equations = []
    m = indicator_var.container

    slack_var = m.addVariable(domain=expr.domain, type="positive")
    slack_eq = m.addEquation(
        domain=expr.domain, definition=(expr.left - slack_var <= expr.right)
    )
    equations.append(slack_eq)

    expr_domain = ... if len(expr.domain) == 0 else [*expr.domain]

    sos_dim = gp.math._generate_dims(m, [2])[0]
    sos1_var = m.addVariable(domain=[*expr.domain, sos_dim], type="sos1")
    sos1_eq_1 = m.addEquation(domain=expr.domain)
    if indicator_val == 1:
        sos1_eq_1[...] = (
            sos1_var[[*expr.domain, "0"]] == indicator_var[expr_domain]
        )
    else:
        sos1_eq_1[...] = (
            sos1_var[[*expr.domain, "0"]] == 1 - indicator_var[expr_domain]
        )
    equations.append(sos1_eq_1)

    sos1_eq_2 = m.addEquation(domain=expr.domain)
    sos1_eq_2[...] = sos1_var[[*expr.domain, "1"]] == slack_var[expr_domain]
    equations.append(sos1_eq_2)

    return equations


def _generate_ray(
    container: gp.Container, domain: typing.Sequence[gp.Set]
) -> tuple[gp.Variable, gp.Variable, list[gp.Equation]]:
    # if b_var == 0 => x_var = 0 o.w x_var >= 0
    # effectively x_var <= bigM * b_var without bigM
    x_var = container.addVariable(domain=domain, type="positive")
    b_var = container.addVariable(domain=domain, type="binary")
    eqs = _indicator(b_var, 0, x_var <= 0)
    return x_var, b_var, eqs


def points_to_intervals(
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
    discontinuous_points: typing.Sequence[int],
) -> list[tuple[int | float, int | float, int | float, int | float]]:
    result: list[
        tuple[int | float, int | float, int | float, int | float]
    ] = []
    finished_at_disc = True
    for i in range(len(x_points) - 1):
        x1 = x_points[i]
        x2 = x_points[i + 1]
        y1 = y_points[i]
        y2 = y_points[i + 1]

        if i in discontinuous_points:
            if finished_at_disc:
                result.append((x1, x1, 0, y1))

            finished_at_disc = True
        else:
            finished_at_disc = False
            slope = (y2 - y1) / (x2 - x1)
            offset = y1 - (slope * x1)
            result.append((x1, x2, slope, offset))

    if finished_at_disc:
        result.append((x2, x2, 0, y2))

    return result


def _get_end_slopes(
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
) -> tuple[float, float]:
    if x_points[-1] != x_points[-2]:
        m_pos = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
    else:
        m_pos = 0

    if x_points[0] != x_points[1]:
        m_neg = (y_points[0] - y_points[1]) / (x_points[0] - x_points[1])
    else:
        m_neg = 0

    return m_neg, m_pos


def pwl_interval_formulation(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
    bound_left: bool = True,
    bound_right: bool = True,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    This function implements a piecewise linear function using the intervals formulation.
    Given an input (independent) variable `input_x`, along with the defining `x_points`
    and corresponding `y_points` of the piecewise function, it constructs the dependent
    variable `y` and formulates the equations necessary to define the function.

    Here is the interval formulation:

    .. math::
        \\lambda_i \\geq b_i * LB_i \\quad \\forall{i}

        \\lambda_i \\leq b_i * UB_i \\quad \\forall{i}

        \\sum_{i}{b_i} = 1

        x = \\sum_{i}{\\lambda_i}

        y = \\sum_{i}{(\\lambda_i * slope_i) + (b_i * offset_i) }

        b_i \\in \\{0, 1\\} \\quad \\forall{i}

    The implementation handles discontinuities in the function. To represent a
    discontinuity at a specific point `x_i`, include `x_i` twice in the `x_points`
    array with corresponding values in `y_points`. For example, if `x_points` =
    [1, 3, 3, 5] and `y_points` = [10, 30, 50, 70], the function allows y to take
    either 30 or 50 when x = 3. Note that discontinuities introduce additional
    binary variables.

    It is possible to disallow a specific range by including `None` in both
    `x_points` and the corresponding `y_points`. For example, with
    `x_points` = `[1, 3, None, 5, 7]` and `y_points` = `[10, 35, None, -20, 40]`,
    the range between 3 and 5 is disallowed for `input_x`.

    However, `x_points` cannot start or end with a `None` value, and a `None`
    value cannot be followed by another `None`. Additionally, if `x_i` is `None`,
    then `y_i` must also be `None`. Similar to the discontinuities, disallowed
    ranges always introduce additional binary variables.

    The input variable `input_x` is restricted to the range defined by
    `x_points` unless `bound_left` or `bound_right` is set to False. Setting
    either to True, creates SOS1 type of variables. When `input_x` is not bound,
    you can assume as if the first and/or the last line segments are extended.

    Returns the dependent variable `y` and the equations required to model the
    piecewise linear relationship.

    Parameters
    ----------
    x : gp.Variable
        Independent variable of the piecewise linear function
    x_points: typing.Sequence[int | float]
        Break points of the piecewise linear function in the x-axis
    y_points: typing.Sequence[int | float]
        Break points of the piecewise linear function in the y-axis
    bound_left: bool = True
        If input_x should be limited to start from x_points[0]
    bound_right: bool = True
        If input_x should be limited to end at x_points[-1]

    Returns
    -------
    tuple[gp.Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.formulations import pwl_interval_formulation
    >>> m = Container()
    >>> x = Variable(m, "x")
    >>> y, eqs = pwl_interval_formulation(x, [-1, 4, 10, 10, 20], [-2, 8, 15, 17, 37])

    """

    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if not isinstance(bound_left, bool):
        raise ValidationError("bound_left is expected to be a boolean")

    if not isinstance(bound_right, bool):
        raise ValidationError("bound_right is expected to be a boolean")

    x_points, y_points, discontinuous_indices, none_indices = _check_points(
        x_points, y_points
    )
    combined_indices = list({*discontinuous_indices, *none_indices})
    equations = []

    intervals = points_to_intervals(x_points, y_points, combined_indices)
    lowerbounds_input = []
    upperbounds_input = []
    slopes_input = []
    offsets_input = []
    for i, (lb, ub, slope, offset) in enumerate(intervals):
        lowerbounds_input.append((str(i), lb))
        upperbounds_input.append((str(i), ub))
        slopes_input.append((str(i), slope))
        offsets_input.append((str(i), offset))

    input_domain = input_x.domain
    m = input_x.container

    J = gp.math._generate_dims(m, [len(intervals)])[0]
    J = gp.formulations.nn.utils._next_domains([J], input_domain)[0]

    lowerbounds = m.addParameter(domain=[J], records=lowerbounds_input)
    upperbounds = m.addParameter(domain=[J], records=upperbounds_input)
    slopes = m.addParameter(domain=[J], records=slopes_input)
    offsets = m.addParameter(domain=[J], records=offsets_input)
    bin_var = m.addVariable(domain=[*input_domain, J], type="binary")

    lambda_var = m.addVariable(domain=[*input_domain, J])

    set_lambda_lowerbound = m.addEquation(domain=lambda_var.domain)
    set_lambda_lowerbound[...] = lowerbounds * bin_var <= lambda_var
    equations.append(set_lambda_lowerbound)

    set_lambda_upperbound = m.addEquation(domain=lambda_var.domain)
    set_lambda_upperbound[...] = upperbounds * bin_var >= lambda_var
    equations.append(set_lambda_upperbound)

    out_y = m.addVariable(domain=input_domain)

    x_term = 0
    y_term = 0
    pick_one_term = 0

    if bound_left is False or bound_right is False:
        m_neg, m_pos = _get_end_slopes(x_points, y_points)

    if bound_left:
        out_y.lo[...] = min(y_points)
    else:
        x_neg_inf, b_neg_inf, eqs_neg_inf = _generate_ray(m, input_domain)
        equations.extend(eqs_neg_inf)

        x_term += -x_neg_inf + (b_neg_inf * x_points[0])
        y_term += -(m_neg * x_neg_inf) + (b_neg_inf * y_points[0])
        pick_one_term += b_neg_inf

    if bound_right:
        out_y.up[...] = max(y_points)
    else:
        x_pos_inf, b_pos_inf, eqs_pos_inf = _generate_ray(m, input_domain)
        equations.extend(eqs_pos_inf)

        x_term += x_pos_inf + (b_pos_inf * x_points[-1])
        y_term += (m_pos * x_pos_inf) + (b_pos_inf * y_points[-1])
        pick_one_term += b_pos_inf

    pick_one = m.addEquation(domain=input_domain)
    pick_one[...] = gp.Sum(J, bin_var) + pick_one_term == 1
    equations.append(pick_one)

    set_x = m.addEquation(domain=input_domain)
    set_x[...] = input_x == gp.Sum(J, lambda_var) + x_term
    equations.append(set_x)

    set_y = m.addEquation(domain=input_domain)
    set_y[...] = (
        out_y
        == gp.Sum(J, lambda_var * slopes)
        + gp.Sum(J, bin_var * offsets)
        + y_term
    )
    equations.append(set_y)

    return out_y, equations


def pwl_convexity_formulation(
    input_x: gp.Variable,
    x_points: typing.Sequence[int | float],
    y_points: typing.Sequence[int | float],
    using: typing.Literal["binary", "sos2"] = "binary",
    bound_left: bool = True,
    bound_right: bool = True,
) -> tuple[gp.Variable, list[gp.Equation]]:
    """
    This function implements a piecewise linear function using the convexity formulation.
    Given an input (independent) variable `input_x`, along with the defining `x_points`
    and corresponding `y_points` of the piecewise function, it constructs the dependent
    variable `y` and formulates the equations necessary to define the function.

    Here is the convexity formulation:

    .. math::
        x = \\sum_{i}{x\\_points_i * \\lambda_i}

        y = \\sum_{i}{y\\_points_i * \\lambda_i}

        \\sum_{i}{\\lambda_i} = 1

        \\lambda_i \\in SOS2


    By default, SOS2 variables are implemented using binary variables.
    See
    `Modeling disjunctive constraints with a logarithmic number of binary variables and constraints
    <https://link.springer.com/article/10.1007/s10107-009-0295-4>`_
    . However, you can switch to SOS2 (Special Ordered Set Type 2) by setting the
    `using` parameter to `"sos2"`.

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
    `x_points` unless `bound_left` or `bound_right` is set to False. Setting
    either to True, creates SOS1 type of variables. When `input_x` is not bound,
    you can assume as if the first and/or the last line segments are extended.

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
    bound_left: bool = True
        If input_x should be limited to start from x_points[0]
    bound_right: bool = True
        If input_x should be limited to end at x_points[-1]

    Returns
    -------
    tuple[gp.Variable, list[Equation]]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.formulations import pwl_convexity_formulation
    >>> m = Container()
    >>> x = Variable(m, "x")
    >>> y, eqs = pwl_convexity_formulation(x, [-1, 4, 10, 10, 20], [-2, 8, 15, 17, 37])

    """
    if using not in {"binary", "sos2"}:
        raise ValidationError(
            "Invalid value for the using argument."
            "Possible values are 'binary' and 'sos2'"
        )

    if not isinstance(input_x, gp.Variable):
        raise ValidationError("input_x is expected to be a Variable")

    if not isinstance(bound_left, bool):
        raise ValidationError("bound_left is expected to be a boolean")

    if not isinstance(bound_right, bool):
        raise ValidationError("bound_right is expected to be a boolean")

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

    x_term = 0
    y_term = 0

    if bound_left is False or bound_right is False:
        m_neg, m_pos = _get_end_slopes(x_points, y_points)

    if bound_left:
        out_y.lo[...] = min(y_points)
    else:
        x_neg_inf, b_neg_inf, eqs_neg_inf = _generate_ray(m, input_domain)
        equations.extend(eqs_neg_inf)

        limit_b_neg_inf = m.addEquation(domain=b_neg_inf.domain)
        limit_b_neg_inf[...] = b_neg_inf <= lambda_var[[*input_domain, "0"]]
        equations.append(limit_b_neg_inf)

        x_term += -x_neg_inf
        y_term += -(m_neg * x_neg_inf)

    if bound_right:
        out_y.up[...] = max(y_points)
    else:
        x_pos_inf, b_pos_inf, eqs_pos_inf = _generate_ray(m, input_domain)
        equations.extend(eqs_pos_inf)

        limit_b_pos_inf = m.addEquation(domain=b_pos_inf.domain)
        last = str(len(J) - 1)
        limit_b_pos_inf[...] = b_pos_inf <= lambda_var[[*input_domain, last]]
        equations.append(limit_b_pos_inf)

        x_term += x_pos_inf
        y_term += m_pos * x_pos_inf

    lambda_sum = m.addEquation(domain=input_x.domain)
    lambda_sum[...] = gp.Sum(J, lambda_var) == 1
    equations.append(lambda_sum)

    set_x = m.addEquation(domain=input_x.domain)
    set_x[...] = input_x == gp.Sum(J, x_par * lambda_var) + x_term
    equations.append(set_x)

    set_y = m.addEquation(domain=input_x.domain)
    set_y[...] = out_y == gp.Sum(J, y_par * lambda_var) + y_term
    equations.append(set_y)

    if using == "binary":
        extra_eqs = _enforce_sos2_with_binary(lambda_var)
        equations.extend(extra_eqs)

    if len(combined_indices) > 0:
        extra_eqs = _enforce_discontinuity(lambda_var, combined_indices)
        equations.extend(extra_eqs)

    return out_y, equations
