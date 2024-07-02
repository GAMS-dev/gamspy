from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def cos(x: int | float | Symbol) -> Expression:
    """
    Cosine of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import cos
    >>> import numpy as np
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = cos(np.pi)
    >>> r.toValue()
    -1.0

    """
    return expression.Expression(None, MathOp("cos", (x,)), None)


def cosh(x: int | float | Symbol) -> Expression:
    """
    Hyperbolic cosine of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import cosh
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = cosh(0)
    >>> r.toValue()
    1.0

    """
    return expression.Expression(None, MathOp("cosh", (x,)), None)


def sin(x: float | Symbol) -> Expression:
    """
    Sine of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sin
    >>> import numpy as np
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = sin(np.pi/2)
    >>> r.toValue()
    1.0

    """
    return expression.Expression(None, MathOp("sin", (x,)), None)


def sinh(x: float | Symbol) -> Expression:
    """
    Hyperbolic sine of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sinh
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = sinh(0)
    >>> r.toValue()
    0.0

    """
    return expression.Expression(None, MathOp("sinh", (x,)), None)


def tan(x: float | Symbol) -> Expression:
    """
    Tangent of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import tan
    >>> import numpy as np
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = tan(np.pi/4)
    >>> round(r.toValue(), 2)
    1.0

    """
    return expression.Expression(None, MathOp("tan", (x,)), None)


def tanh(x: float | Symbol) -> Expression:
    """
    Hyperbolic tangent of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import tanh
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = tanh(0)
    >>> r.toValue()
    0.0

    """
    return expression.Expression(None, MathOp("tanh", (x,)), None)


def acos(x: float | Symbol) -> Expression:
    """
    Inverse cosine of ``x``.

    Returns
    -------
    Expresion | float

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import acos
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = acos(1)
    >>> r.toValue()
    0.0

    """
    return expression.Expression(None, MathOp("arccos", (x,)), None)


def asin(x: float | Symbol) -> Expression:
    """
    Inver sinus of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import asin
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = asin(0)
    >>> r.toValue()
    0.0

    """
    return expression.Expression(None, MathOp("arcsin", (x,)), None)


def atan(x: float | Symbol) -> Expression:
    """
    Inverse tangent of ``x``.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import atan
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = atan(0)
    >>> r.toValue()
    0.0

    """
    return expression.Expression(None, MathOp("arctan", (x,)), None)


def atan2(y: int | float | Symbol, x: int | float | Symbol) -> Expression:
    """
    Four-quadrant arctan function yielding ``arctan(y/x)``, which is the angle the vector ``(x,y)`` makes with ``(1,0)`` in radians.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import atan2
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = atan2(1,1)
    >>> r.toValue()
    0.7853981633974483

    """
    return expression.Expression(None, MathOp("arctan2", (y, x)), None)
