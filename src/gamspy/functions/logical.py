from typing import TYPE_CHECKING
import gamspy._algebra._expression as expression
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy._algebra._expression import Expression


def ifthen(
    condition: "Expression", yes_return: float, no_return: float
) -> "Expression":
    condition_str = condition.gamsRepr()
    condition_str = utils._replaceEqualitySigns(condition_str)

    ifthen_str = f"ifthen({condition_str}, {yes_return}, {no_return})"
    return expression.Expression(ifthen_str, "", "")
