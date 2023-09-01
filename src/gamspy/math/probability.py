import gamspy._algebra.expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def centropy(
    x: "Symbol",
    y: "Symbol",
    z: float = 1e-20,
) -> "Expression":
    """
    Cross-entropy. x . ln((x + z) / (y + z)

    Parameters
    ----------
    x : float | Operable
    y : float | Operable
    z : float, optional

    Returns
    -------
    Expression

    Raises
    ------
    ValueError
        if z is smaller than 0
    """
    if z < 0:
        raise ValueError("z must be greater than or equal to 0")

    return expression.Expression(
        "centropy(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
    )


def uniform(lower_bound: float, upper_bound: float) -> "Expression":
    """
    Generates a random number from the uniform distribution between
    lower_bound and higher_bound

    Parameters
    ----------
    lower_bound : float
    upper_bound : float

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "uniform(", f"{lower_bound},{upper_bound}", ")"
    )


def uniformInt(lower_bound: float, upper_bound: float) -> "Expression":
    """
    Generates an integer random number from the discrete uniform distribution
    whose outcomes are the integers between lower_bound and higher_bound.

    Parameters
    ----------
    lower_bound : float
    upper_bound : float

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "uniformInt(", f"{lower_bound},{upper_bound}", ")"
    )


def normal(mean: Union[int, float], dev: Union[int, float]) -> "Expression":
    """
    Generate a random number from the normal distribution with mean `mean`
    and `standard deviation` dev.

    Parameters
    ----------
    mean : int | float
    dev : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("normal(", f"{mean},{dev}", ")")
