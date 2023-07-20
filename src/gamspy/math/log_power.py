import math
import gamspy._algebra._expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gamspy._algebra._expression import Expression


def exp(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.exp(value)
    return expression.Expression("exp(", value.gamsRepr(), ")")


def log(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.log(value)
    return expression.Expression("log(", value.gamsRepr(), ")")


def log2(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.log2(value)
    return expression.Expression("log2(", value.gamsRepr(), ")")


def log10(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.log10(value)
    return expression.Expression("log10(", value.gamsRepr(), ")")


def power(base, exponent):
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
