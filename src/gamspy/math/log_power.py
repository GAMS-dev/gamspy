import math
import gamspy._algebra._expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra._expression import Expression
    from gamspy._algebra._operable import Operable


def exp(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Exponential of x (i.e. e^x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.exp(x)
    return expression.Expression("exp(", x.gamsRepr(), ")")


def log(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.log(x)
    return expression.Expression("log(", x.gamsRepr(), ")")


def log2(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Binary logarithm (i.e. logarithm base 2 of x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.log2(x)
    return expression.Expression("log2(", x.gamsRepr(), ")")


def log10(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Common logarithm (i.e. logarithm base 10 of x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.log10(x)
    return expression.Expression("log10(", x.gamsRepr(), ")")


def power(
    base: Union[float, "Operable"], exponent: Union[float, "Operable"]
) -> Union["Expression", float]:
    """
    Base to the exponent power (i.e. base ^ exponent)

    Parameters
    ----------
    base : float | Operable
    exponent : float | Operable

    Returns
    -------
    Expression | float
    """
    if isinstance(base, (int, float)) and isinstance(exponent, (int, float)):
        return math.pow(base, exponent)

    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression("power(", f"{base_str},{exponent_str}", ")")
