import math
import gamspy._algebra.expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def cos(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Cosine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.cos(x)
    return expression.Expression("cos(", x.gamsRepr(), ")")


def cosh(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Hyperbolic cosine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.cosh(x)
    return expression.Expression("cosh(", x.gamsRepr(), ")")


def sin(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Sine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.sin(x)
    return expression.Expression("sin(", x.gamsRepr(), ")")


def sinh(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Hyperbolic sine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.sinh(x)
    return expression.Expression("sinh(", x.gamsRepr(), ")")


def tan(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Tangent of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.tan(x)
    return expression.Expression("tan(", x.gamsRepr(), ")")


def tanh(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Hyperbolic tangent of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.tanh(x)
    return expression.Expression("tanh(", x.gamsRepr(), ")")


def acos(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Inverse cosine of x.

    Returns
    -------
    Expresion | float
    """
    if isinstance(x, (int, float)):
        return math.acos(x)
    return expression.Expression("arccos(", x.gamsRepr(), ")")


def asin(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Inver sinus of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.asin(x)
    return expression.Expression("arcsin(", x.gamsRepr(), ")")


def atan(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Inverse tangent of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.atan(x)
    return expression.Expression("arctan(", x.gamsRepr(), ")")


def atan2(
    y: Union[int, float, "Symbol"], x: Union[int, float, "Symbol"]
) -> Union["Expression", float]:
    """
    Four-quadrant arctan function

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return math.atan2(y, x)

    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    y_str = str(y) if isinstance(y, (int, float)) else y.gamsRepr()

    return expression.Expression("arctan(", f"{y_str}, {x_str}", ")")
