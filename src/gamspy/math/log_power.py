from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def exp(x: float | Symbol) -> Expression:
    """
    Exponential of ``x`` (i.e. ``e^x``)

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import exp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = exp(a)
    >>> b.toValue()
    44.701184493300815

    """
    return expression.Expression(None, MathOp("exp", (x,)), None)


def log(x: int | float | Symbol) -> Expression:
    """
    Natural logarithm of ``x`` (i.e. logarithm base ``e`` of ``x``)

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log(a)
    >>> b.toValue()
    1.33500106673234

    """
    return expression.Expression(None, MathOp("log", (x,)), None)


def log_beta(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Log beta function (i.e. ``log(B(x, y)``)

    Parameters
    ----------
    x : int | float | Symbol
    y : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log_beta
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log_beta(a,5)
    >>> b.toValue()
    -5.45446741772822

    """
    return expression.Expression(None, MathOp("logBeta", (x, y)), None)


def log_gamma(x: int | float | Symbol) -> Expression:
    """
    Log gamma function of ``x``

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log_gamma
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log_gamma(a)

    """
    return expression.Expression(None, MathOp("logGamma", (x,)), None)


def logit(x: int | float | Symbol) -> Expression:
    """
    Logit Transformation (i.e. ``log(x / (1 - x))``) for ``x`` in ``(0, 1)``

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import logit
    >>> m = Container()
    >>> a = Parameter(m, "a", records=0.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = logit(a)
    >>> b.toValue()
    1.3862943611198908

    """
    return expression.Expression(None, MathOp("logit", (x,)), None)


def log2(x: float | Symbol) -> Expression:
    """
    Binary logarithm (i.e. logarithm base ``2`` of ``x``)

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log2
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log2(a)
    >>> b.toValue()
    1.9259994185562224

    """
    return expression.Expression(None, MathOp("log2", (x,)), None)


def log10(x: float | Symbol) -> Expression:
    """
    Common logarithm (i.e. logarithm base ``10`` of ``x``)

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log10
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log10(a)
    >>> b.toValue()
    0.5797835966168101

    """
    return expression.Expression(None, MathOp("log10", (x,)), None)


def power(base: float | Symbol, exponent: float | Symbol) -> Expression:
    """
    Base to the exponent power (i.e. ``base ^ exponent``)

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import power
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = power(a, 3)
    >>> b.toValue()
    54.87199999999999

    """
    return expression.Expression(None, MathOp("power", (base, exponent)), None)


def cv_power(base: float, exponent: float | Symbol) -> Expression:
    """
    Real power (i.e. ``base ^ exponent`` where ``base >= 0``; error for ``base < 0``)

    Parameters
    ----------
    base : float
    exponent : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import cv_power
    >>> m = Container()
    >>> a = Parameter(m, "a", records=4)
    >>> b = Parameter(m, "b")
    >>> b[...] = cv_power(3, a)

    """
    if not isinstance(base, (float, int)):
        raise ValueError("Base must be a number")

    if base < 0:
        raise ValueError("Base must be greater than or equal to 0")

    return expression.Expression(
        None, MathOp("cvPower", (base, exponent)), None
    )


def rpower(base: float | Symbol, exponent: float | Symbol):
    """
    Returns ``x^y`` for ``x > 0`` and also for ``x = 0`` and restricted values of ``y`` (Error if ``x < 0``)

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter, Set
    >>> from gamspy.math import rpower
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 3.8)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = rpower(a[i], 3)
    >>> b.toList()
    [('i1', 54.87199999999999)]

    """
    return expression.Expression(
        None, MathOp("rPower", (base, exponent)), None
    )


def sign_power(base: float | Symbol, exponent: float):
    """
    Signed power: ``sign(base) * |base|^exponent``, for ``exponent > 0``

    Parameters
    ----------
    base : float | Symbol
    exponent : float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter, Set
    >>> from gamspy.math import sign_power
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 3.8)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sign_power(a[i], 5)
    >>> b.toList()
    [('i1', 792.3516799999994)]

    """
    if not isinstance(exponent, (float, int)):
        raise ValueError("Exponent must be a number")

    if exponent <= 0:
        raise ValueError("Exponent must be greater than 0")

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

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sllog10
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = sllog10(a)
    >>> b.toValue()
    0.5797835966168101

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

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sqlog10
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = sqlog10(a)
    >>> b.toValue()
    0.5797835966168101

    """
    return expression.Expression(None, MathOp("sqlog10", (x, S)), None)


def vc_power(base: float | Symbol, exponent: float | Symbol):
    """
    Returns ``x^y`` for ``x >= 0`` (error for ``x < 0``)

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import vc_power
    >>> m = Container()
    >>> a = Parameter(m, "a", records=4)
    >>> b = Parameter(m, "b")
    >>> b[...] = vc_power(a, 3)

    """
    return expression.Expression(
        None, MathOp("vcPower", (base, exponent)), None
    )


def sqr(x: float | Symbol) -> Expression:
    """
    Square of ``x`` (i.e. ``x^2``)

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sqr
    >>> m = Container()
    >>> a = Parameter(m, "a", records=4)
    >>> b = Parameter(m, "b")
    >>> b[...] = sqr(a)
    >>> b.toValue()
    16.0

    """
    return expression.Expression(None, MathOp("sqr", (x,)), None)
