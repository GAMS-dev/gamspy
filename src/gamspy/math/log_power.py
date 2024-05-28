from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def exp(x: float | Symbol) -> Expression:
    """
    Exponential of x (i.e. e^x)

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("exp", (x,)), None)


def log(x: int | float | Symbol) -> Expression:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("log", (x,)), None)


def log_beta(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Log beta function

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("logBeta", (x, y)), None)


def log_gamma(x: int | float | Symbol) -> Expression:
    """
    Log gamma function

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("logGamma", (x,)), None)


def logit(x: int | float | Symbol) -> Expression:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("logit", (x,)), None)


def log2(x: float | Symbol) -> Expression:
    """
    Binary logarithm (i.e. logarithm base 2 of x)

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("log2", (x,)), None)


def log10(x: float | Symbol) -> Expression:
    """
    Common logarithm (i.e. logarithm base 10 of x)

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("log10", (x,)), None)


def power(base: float | Symbol, exponent: float | Symbol) -> Expression:
    """
    Base to the exponent power (i.e. base ^ exponent)

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("power", (base, exponent)), None)


def cv_power(base: float | Symbol, exponent: float | Symbol) -> Expression:
    """
    Real power (i.e. base ^ exponent where X >= 0)

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        None, MathOp("cvPower", (base, exponent)), None
    )


def rpower(base: float | Symbol, exponent: float | Symbol):
    """
    Returns x^y for x > 0 and also for x = 0 and restricted values of y

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        None, MathOp("rPower", (base, exponent)), None
    )


def sign_power(base: float | Symbol, exponent: float | Symbol):
    """
    Signed power for y > 0.

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        None, MathOp("signPower", (base, exponent)), None
    )


def sllog10(x: int | float | Symbol, S: int | float = 1.0e-150) -> Expression:
    """
    Smooth (linear) logarithm base 10

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 1.0e-150

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sllog10", (x, S)), None)


def sqlog10(x: int | float | Symbol, S: int | float = 1.0e-150) -> Expression:
    """
    Smooth (quadratic) logarithm base 10

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 1.0e-150

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sqlog10", (x, S)), None)


def vc_power(base: float | Symbol, exponent: float | Symbol):
    """
    Returns x^y for x >= 0

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        None, MathOp("vcPower", (base, exponent)), None
    )


def sqr(x: float | Symbol) -> Expression:
    """
    Square of x

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sqr", (x,)), None)
