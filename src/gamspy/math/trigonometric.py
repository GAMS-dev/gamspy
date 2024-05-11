from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def cos(x: int | float | Symbol) -> Expression:
    """
    Cosine of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("cos", (x,)), None)


def cosh(x: int | float | Symbol) -> Expression:
    """
    Hyperbolic cosine of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("cosh", (x,)), None)


def sin(x: float | Symbol) -> Expression:
    """
    Sine of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sin", (x,)), None)


def sinh(x: float | Symbol) -> Expression:
    """
    Hyperbolic sine of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sinh", (x,)), None)


def tan(x: float | Symbol) -> Expression:
    """
    Tangent of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("tan", (x,)), None)


def tanh(x: float | Symbol) -> Expression:
    """
    Hyperbolic tangent of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("tanh", (x,)), None)


def acos(x: float | Symbol) -> Expression:
    """
    Inverse cosine of x.

    Returns
    -------
    Expresion | float
    """
    return expression.Expression(None, MathOp("arccos", (x,)), None)


def asin(x: float | Symbol) -> Expression:
    """
    Inver sinus of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("arcsin", (x,)), None)


def atan(x: float | Symbol) -> Expression:
    """
    Inverse tangent of x.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("arctan", (x,)), None)


def atan2(y: int | float | Symbol, x: int | float | Symbol) -> Expression:
    """
    Four-quadrant arctan function

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("arctan2", (y, x)), None)
