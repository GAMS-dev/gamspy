from typing import Union, TYPE_CHECKING
import gamspy._algebra.expression as expression
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression


def ifthen(
    condition: "Expression",
    yes_return: Union[float, "Expression"],
    no_return: Union[float, "Expression"],
) -> "Expression":
    """
    If the logical condition is true, the function returns iftrue,
    else it returns else

    Parameters
    ----------
    condition : Expression
    yes_return : float | Expression
    no_return : float | Expression

    Returns
    -------
    Expression

    Examples
    --------
    >>> x = ifthen(tt == 2, 3, 4 + y)
    """
    condition_str = condition.gamsRepr()
    condition_str = utils._replaceEqualitySigns(condition_str)

    ifthen_str = f"ifthen({condition_str}, {yes_return}, {no_return})"
    return expression.Expression(ifthen_str, "", "")
