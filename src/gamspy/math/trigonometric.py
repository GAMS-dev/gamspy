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

from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
from gamspy.math.misc import _stringify

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def cos(x: Union[int, float, Symbol]) -> Expression:
    """
    Cosine of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"cos({x_str})", None)


def cosh(x: Union[int, float, Symbol]) -> Expression:
    """
    Hyperbolic cosine of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"cosh({x_str})", None)


def sin(x: Union[float, Symbol]) -> Expression:
    """
    Sine of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"sin({x_str})", None)


def sinh(x: Union[float, Symbol]) -> Expression:
    """
    Hyperbolic sine of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"sinh({x_str})", None)


def tan(x: Union[float, Symbol]) -> Expression:
    """
    Tangent of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"tan({x_str})", None)


def tanh(x: Union[float, Symbol]) -> Expression:
    """
    Hyperbolic tangent of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"tanh({x_str})", None)


def acos(x: Union[float, Symbol]) -> Expression:
    """
    Inverse cosine of x.

    Returns
    -------
    Expresion | float
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"arccos({x_str})", None)


def asin(x: Union[float, Symbol]) -> Expression:
    """
    Inver sinus of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"arcsin({x_str})", None)


def atan(x: Union[float, Symbol]) -> Expression:
    """
    Inverse tangent of x.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"arctan({x_str})", None)


def atan2(
    y: Union[int, float, Symbol], x: Union[int, float, Symbol]
) -> Expression:
    """
    Four-quadrant arctan function

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    y_str = _stringify(y)

    return expression.Expression(None, f"arctan2({y_str},{x_str})", None)
