import math
import gamspy._algebra.expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def cos(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Cosine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.cos(x)
    return expression.Expression("cos(", x.gamsRepr(), ")")


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
