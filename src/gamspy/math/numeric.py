import math
import gamspy._algebra._expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gamspy._algebra._expression import Expression


def abs(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.fabs(value)
    return expression.Expression("abs(", value.gamsRepr(), ")")


def ceil(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.ceil(value)
    return expression.Expression("ceil(", value.gamsRepr(), ")")


def floor(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.floor(value)
    return expression.Expression("floor(", value.gamsRepr(), ")")


def min(*values) -> "Expression":
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("min(", values_str, ")")


def max(*values) -> "Expression":
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("max(", values_str, ")")


def mod(dividend, divider) -> Union["Expression", int, float]:
    if isinstance(dividend, (int, float)) and isinstance(
        divider, (int, float)
    ):
        return dividend % divider

    dividend_str = (
        str(dividend)
        if isinstance(dividend, (int, float))
        else dividend.gamsRepr()
    )
    divider_str = (
        str(divider)
        if isinstance(divider, (int, float))
        else divider.gamsRepr()
    )
    return expression.Expression("mod(" + dividend_str, ",", divider_str + ")")


def Round(value, decimal: int = 0) -> "Expression":
    return expression.Expression(
        "round(", value.gamsRepr() + f", {decimal}", ")"
    )


def sign(value) -> "Expression":
    return expression.Expression("sign(", value.gamsRepr(), ")")


def sqrt(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.sqrt(value)
    return expression.Expression("sqrt(", value.gamsRepr(), ")")
