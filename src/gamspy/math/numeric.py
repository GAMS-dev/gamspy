import math
import gamspy._algebra.expression as expression
from typing import Union, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def abs(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Absolute value of x (i.e. ``|x|``)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.fabs(x)
    return expression.Expression("abs(", x.gamsRepr(), ")")


def ceil(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    The smallest integer greater than or equal to x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.ceil(x)
    return expression.Expression("ceil(", x.gamsRepr(), ")")


def div(
    dividend: Union[int, float, "Symbol"], divisor: Union[int, float, "Symbol"]
) -> Union["Expression", float]:
    """
    Dividing operation

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression | float
    """
    if isinstance(dividend, (int, float)) and isinstance(
        divisor, (int, float)
    ):
        return dividend / divisor

    dividend_str = (
        str(dividend)
        if isinstance(dividend, (int, float))
        else dividend.gamsRepr()
    )
    divisor_str = (
        str(divisor)
        if isinstance(divisor, (int, float))
        else divisor.gamsRepr()
    )

    return expression.Expression("div(", f"{dividend_str}, {divisor_str}", ")")


def div0(
    dividend: Union[int, float, "Symbol"], divisor: Union[int, float, "Symbol"]
) -> Union["Expression", float]:
    """
    Dividing operation

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression | float
    """
    if isinstance(dividend, (int, float)) and isinstance(
        divisor, (int, float)
    ):
        return dividend / divisor

    dividend_str = (
        str(dividend)
        if isinstance(dividend, (int, float))
        else dividend.gamsRepr()
    )
    divisor_str = (
        str(divisor)
        if isinstance(divisor, (int, float))
        else divisor.gamsRepr()
    )

    return expression.Expression(
        "div0(", f"{dividend_str}, {divisor_str}", ")"
    )


def dist(
    x1: Union[Tuple[int, float], "Symbol"],
    x2: Union[Tuple[int, float], "Symbol"],
) -> Union["Expression", float]:
    if isinstance(x1, tuple) and isinstance(x2, tuple):
        return math.dist(x1, x2)

    if isinstance(x1, tuple) or isinstance(x2, tuple):
        raise Exception("Both should be a tuple or none")

    return expression.Expression(
        "eDist(", f"{x1.gamsRepr()}, {x2.gamsRepr()}", ")"
    )


def factorial(x: Union[int, "Symbol"]) -> Union["Expression", int]:
    """
    Factorial of x

    Returns
    -------
    Expression | int
    """
    if isinstance(x, int):
        return math.factorial(x)
    return expression.Expression("ceil(", x.gamsRepr(), ")")


def floor(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    The greatest integer less than or equal to x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.floor(x)
    return expression.Expression("floor(", x.gamsRepr(), ")")


def fractional(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Returns the fractional part of x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        fraction, integer = math.modf(x)
        return fraction
    return expression.Expression("frac(", x.gamsRepr(), ")")


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
    x: Union[float, "Symbol"], y: Union[float, "Symbol"]
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


def Round(x: "Symbol", num_decimals: int = 0) -> "Expression":
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


def sign(x: "Symbol") -> "Expression":
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


def slexp():
    ...


def sqexp():
    ...


def sqrt(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Square root of x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.sqrt(x)
    return expression.Expression("sqrt(", x.gamsRepr(), ")")


def trunc():
    ...
