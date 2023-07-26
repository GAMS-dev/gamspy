import math
import gamspy._algebra._expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra._expression import Expression
    from gamspy._algebra._operable import Operable


def cos(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Cosine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.cos(x)
    return expression.Expression("cos(", x.gamsRepr(), ")")


def sin(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Sine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.sin(x)
    return expression.Expression("sin(", x.gamsRepr(), ")")


def acos(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Arccosine of x.

    Returns
    -------
    Expresion | float
    """
    if isinstance(x, (int, float)):
        return math.acos(x)
    return expression.Expression("arccos(", x.gamsRepr(), ")")


def asin(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Arcsine of x.

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.asin(x)
    return expression.Expression("arcsin(", x.gamsRepr(), ")")
