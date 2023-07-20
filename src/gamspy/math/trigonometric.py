import math
import gamspy._algebra._expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gamspy._algebra._expression import Expression


def cos(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.cos(value)
    return expression.Expression("cos(", value.gamsRepr(), ")")


def sin(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.sin(value)
    return expression.Expression("sin(", value.gamsRepr(), ")")


def acos(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.acos(value)
    return expression.Expression("arccos(", value.gamsRepr(), ")")


def asin(value) -> Union["Expression", float]:
    if isinstance(value, (int, float)):
        return math.asin(value)
    return expression.Expression("arcsin(", value.gamsRepr(), ")")
