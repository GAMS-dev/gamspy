import math
import gamspy._algebra.expression as expression
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def exp(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Exponential of x (i.e. e^x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.exp(x)
    return expression.Expression("exp(", x.gamsRepr(), ")")


def log(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.log(x)
    return expression.Expression("log(", x.gamsRepr(), ")")


def log_beta(
    x: Union[int, float, "Symbol"], y: Union[int, float, "Symbol"]
) -> "Expression":
    """
    Log beta function

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    y_str = str(y) if isinstance(y, (int, float)) else y.gamsRepr()
    return expression.Expression("logBeta(", f"{x_str},{y_str}", ")")


def log_gamma(
    x: Union[int, float, "Symbol"], y: Union[int, float, "Symbol"]
) -> "Expression":
    """
    Log gamma function

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    y_str = str(y) if isinstance(y, (int, float)) else y.gamsRepr()
    return expression.Expression("logGamma(", f"{x_str},{y_str}", ")")


def logit(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression | float
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("logit(", x_str, ")")


def log2(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Binary logarithm (i.e. logarithm base 2 of x)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.log2(x)
    return expression.Expression("log2(", x.gamsRepr(), ")")


def log10(x: Union[float, "Symbol"]) -> Union["Expression", float]:
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
    base: Union[float, "Symbol"], exponent: Union[float, "Symbol"]
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


def cv_power(
    base: Union[float, "Symbol"], exponent: Union[float, "Symbol"]
) -> Union["Expression", float]:
    """
    Real power (i.e. base ^ exponent where X >= 0)

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
    return expression.Expression("cvPower(", f"{base_str},{exponent_str}", ")")


def rpower():
    ...


def sign_power():
    ...


def sllog10():
    ...


def sqlog10():
    ...


def vc_power():
    ...


def sqr(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Square of x

    Parameters
    ----------
    x : float | Operable

    Returns
    -------
    Expression | float
    """
    return power(x, 2)
