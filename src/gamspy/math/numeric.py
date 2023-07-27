import math
import gamspy._algebra._expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra._expression import Expression
    from gamspy._algebra._operable import Operable


def abs(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Absolute value of x (i.e. |x|)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.fabs(x)
    return expression.Expression("abs(", x.gamsRepr(), ")")


def ceil(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    The smallest integer greater than or equal to x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.ceil(x)
    return expression.Expression("ceil(", x.gamsRepr(), ")")


def floor(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    The greatest integer less than or equal to x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.floor(x)
    return expression.Expression("floor(", x.gamsRepr(), ")")


def min(*values) -> "Expression":
    """
    Minimum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("min(", values_str, ")")


def max(*values) -> "Expression":
    """
    Maximum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("max(", values_str, ")")


def mod(
    x: Union[float, "Operable"], y: Union[float, "Operable"]
) -> Union["Expression", int, float]:
    """
    Remainder of x divided by y.

    Returns
    -------
    Expression | int | float
    """
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x % y

    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    y_str = str(y) if isinstance(y, (int, float)) else y.gamsRepr()
    return expression.Expression("mod(" + x_str, ",", y_str + ")")


def Round(x: "Operable", num_decimals: int = 0) -> "Expression":
    """
    Round x to num_decimals decimal places.

    Parameters
    ----------
    x : Operable
    decimal : int, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "round(", x.gamsRepr() + f", {num_decimals}", ")"
    )


def sign(x: "Operable") -> "Expression":
    """
    Sign of x returns 1 if x > 0, -1 if x < 0, and 0 if x = 0

    Parameters
    ----------
    x : Operable

    Returns
    -------
    Expression
    """
    return expression.Expression("sign(", x.gamsRepr(), ")")


def sqrt(x: Union[float, "Operable"]) -> Union["Expression", float]:
    """
    Square root of x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.sqrt(x)
    return expression.Expression("sqrt(", x.gamsRepr(), ")")
