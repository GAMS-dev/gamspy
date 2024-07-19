#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import gamspy._symbols.implicits as implicits
import gamspy.math
from gamspy._symbols.parameter import Parameter
from gamspy._symbols.variable import Variable
from gamspy.exceptions import ValidationError
from gamspy.math.matrix import next_alias

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Operation


def _get_random_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}".replace("-", "_")


def _get_lb(x: Variable, default_lb: float):
    lb = x.container.addParameter(
        _get_random_name("lb"),
        domain=x.domain,
    )
    lb[...] = x.lo[...]
    lb[...].where[lb[...] == "-inf"] = default_lb
    return lb


def _get_ub(x: Variable, default_ub: float):
    ub = x.container.addParameter(
        _get_random_name("ub"),
        domain=x.domain,
    )
    ub[...] = x.up[...]
    ub[...].where[ub[...] == "inf"] = default_ub
    return ub


def relu_with_binary_var(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
    default_lb: float = -(10**6),
    default_ub: float = 10**6,
):
    """
    Implements the ReLU activation function using binary variables. The ReLU
    function is defined as ReLU(x) = max(x, 0). This implementation **generates**
    one binary variable and one positive variable. The binary variable is
    necessary to represent the mathematical relationship, while the positive
    variable serves as the activation variable. Both the binary and positive
    variables share the same domain as the input.

    The formulation of this function requires having lower and upper bounds
    for the input ``x``. This function utilizes the bounds from the variables
    if provided. If not, it defaults to the bounds defined by ``default_lb``
    and ``default_ub``. Providing tighter and **correct** bounds can enhance
    the quality of linear relaxations.

    Returns activation variable and binary variable in order.

    Adapted from `OMLT <https://github.com/cog-imperial/OMLT/blob/e60563859a66ac5dd3348bf1763de57eec95171e/src/omlt/neuralnet/activations/relu.py#L5>`_

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation
    default_ub : float
    default_lb : float

    Returns
    -------
    tuple[Variable, Variable]

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import relu_with_binary_var
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y, b = relu_with_binary_var(x) # mostly, you can ignore b
    >>> y.type
    'positive'
    >>> b.type
    'binary'
    >>> y.domain # i many activation variables
    [<Set `i` (0x...)>]
    >>> b.domain # i many binary variables
    [<Set `i` (0x...)>]
    """

    domain = x.domain
    sigma = x.container.addVariable(
        _get_random_name("bin"),
        type="binary",
        domain=domain,
    )

    y = x.container.addVariable(
        _get_random_name("y"),
        type="positive",
        domain=domain,
    )

    eq = [
        x.container.addEquation(
            _get_random_name("eq"),
            domain=domain,
        )
        for _ in range(3)
    ]

    # y >= 0 implied by positive variable
    eq[0][...] = y >= x

    if isinstance(x, Variable):
        eq[1][...] = y <= x - (1 - sigma) * _get_lb(x, default_lb)
        eq[2][...] = y <= sigma * _get_ub(x, default_ub)
        y.lo[...] = gamspy.math.Max(0, x.lo[...])
        y.up[...] = gamspy.math.Max(0, x.up[...])
    else:
        eq[1][...] = y <= x - (1 - sigma) * default_lb
        eq[2][...] = y <= sigma * default_ub
        y.lo[...] = 0

    return y, sigma


def relu_with_complementarity_var(
    x: (
        Parameter
        | Variable
        | implicits.ImplicitParameter
        | implicits.ImplicitVariable
        | Expression
        | Operation
    ),
):
    """
    Implements the ReLU activation function using complementarity conditions.
    The ReLU function is defined as ReLU(x) = max(x, 0). This implementation
    **generates** one positive variable, which serves as the activation variable.
    The activation variable shares the same domain as the input. Unlike
    ``relu_with_binary_var``, this function does not require lower and upper
    bounds for the formulation.

    Returns the activation variable.

    Adapted from `OMLT <https://github.com/cog-imperial/OMLT/blob/e60563859a66ac5dd3348bf1763de57eec95171e/src/omlt/neuralnet/activations/relu.py#L85>`_

    Parameters
    ----------
    x : Parameter | Variable | implicits.ImplicitParameter | implicits.ImplicitVariable | Expression | Operation

    Returns
    -------
    Variable

    Examples
    --------
    >>> from gamspy import Container, Variable, Set
    >>> from gamspy.math.activation import relu_with_complementarity_var
    >>> m = Container()
    >>> i = Set(m, "i", records=range(3))
    >>> x = Variable(m, "x", domain=[i])
    >>> y = relu_with_complementarity_var(x)
    >>> y.type
    'positive'
    """
    domain = x.domain

    y = x.container.addVariable(
        _get_random_name("y"),
        type="positive",
        domain=domain,
    )

    eq = [
        x.container.addEquation(
            _get_random_name("eq"),
            domain=domain,
        )
        for _ in range(2)
    ]

    # y >= 0 implied by positive variable
    eq[0][...] = y * (y - x) == 0
    eq[1][...] = y - x >= 0

    if isinstance(x, Variable):
        y.lo[...] = gamspy.math.Max(0, x.lo[...])
        y.up[...] = gamspy.math.Max(0, x.up[...])
    else:
        y.lo[...] = 0

    return y


def log_softmax(x: Variable, dim: int = -1):
    """
    Implements the log_softmax activation function. This function strictly
    requires a GAMSPy Variable, `y = log_softmax(x)`. The ``dim`` parameter
    specifies the **index** of the softmax dimension. If not provided, it
    calculates log_softmax for the last dimension. This function is preferred
    over the :meth:`softmax <gamspy.math.softmax>` function because, when
    the softmax dimension has 20 or fewer elements, it uses the
    :meth:`lse_max <gamspy.math.lse_max>` (log-sum-exp) intrinsic function for
    improved numerical stability which usually leads to faster solve times.

    To learn more about `Log-Sum-Exp trick <https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/>`_ .

    This function is usually combined with Negative Log Likelihood loss for
    classification problems.

    Returns the activation variable.

    Parameters
    ----------
    x : Variable
    dim : int

    Returns
    -------
    Variable

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.math import dim
    >>> from gamspy.math.activation import log_softmax
    >>> m = Container()
    >>> x = Variable(m, "x", domain=dim([500, 10]))
    >>> y = log_softmax(x) # uses LSE because 10 <= 20
    >>> y.domain
    [<Set `DenseDim500_1` (0x...)>, <Set `DenseDim10_1` (0x...)>]
    >>> y2 = log_softmax(x, dim=0) # cannot use LSE because 500 > 20
    """
    if not isinstance(x, Variable):
        raise ValidationError("log_softmax expects a variable")

    if dim < 0:
        dim = len(x.domain) + dim

    sum_domain = next_alias(x.domain[dim])

    y = x.container.addVariable(
        _get_random_name("y"),
        domain=x.domain,
    )

    eq = x.container.addEquation(
        _get_random_name("eq"),
        domain=x.domain,
    )

    if len(sum_domain) != 0 and len(sum_domain) <= 20:
        # Use built-in LSE if possible
        scalars = [rec for rec in sum_domain.records["uni"]]
        variables = []
        for scalar in scalars:
            expr_domain = [*x.domain[:dim], scalar, *x.domain[dim + 1 :]]
            variables.append(x[expr_domain])

        log_sum_exp = gamspy.math.lse_max(*variables)
        eq[...] = y[...] == x - log_sum_exp
    else:
        expr_domain = [
            d if i != dim else sum_domain for (i, d) in enumerate(x.domain)
        ]
        sum_expr = gamspy.Sum(sum_domain, gamspy.math.exp(x[expr_domain]))
        eq[...] = y[...] == x - gamspy.math.log(sum_expr)

    return y


def softmax(x: Variable, dim: int = -1):
    """
    Implements the softmax activation function. This function strictly requires
    a GAMSPy Variable, `y = softmax(x)`. The ``dim`` parameter specifies the
    index of the softmax dimension. If not provided, the softmax is calculated
    for the last dimension. This function is implemented for completeness;
    however, in many cases, you can use :meth:`log_softmax <gamspy.math.log_softmax>`
    for better numerical stability.

    Use :meth:`log_softmax <gamspy.math.log_softmax>` if you need to take the
    logarithm of the softmax function.

    Returns the activation variable.

    Parameters
    ----------
    x : Variable
    dim : int

    Returns
    -------
    Variable

    Examples
    --------
    >>> from gamspy import Container, Variable
    >>> from gamspy.math import dim
    >>> from gamspy.math.activation import softmax
    >>> m = Container()
    >>> x = Variable(m, "x", domain=dim([500, 10]))
    >>> y = softmax(x)
    >>> y.domain
    [<Set `DenseDim500_1` (0x...)>, <Set `DenseDim10_1` (0x...)>]
    """
    if not isinstance(x, Variable):
        raise ValidationError("softmax expects a variable")

    if dim < 0:
        dim = len(x.domain) + dim

    sum_domain = next_alias(x.domain[dim])
    expr_domain = [
        d if i != dim else sum_domain for (i, d) in enumerate(x.domain)
    ]

    y = x.container.addVariable(
        _get_random_name("y"),
        domain=x.domain,
    )

    eq = x.container.addEquation(
        _get_random_name("eq"),
        domain=x.domain,
    )

    sum_expr = gamspy.Sum(sum_domain, gamspy.math.exp(x[expr_domain]))
    eq[...] = y[...] == gamspy.math.exp(x) / sum_expr

    return y
