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

    This function utilizes the bounds from the variables if provided. If not,
    it defaults to the bounds defined by ``default_lb`` and ``default_ub``.
    Providing tighter and **correct** bounds can enhance the quality of linear
    relaxations.

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
    default_lb: float = -(10**6),
    default_ub: float = 10**6,
):
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
