from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operable import Operable
    from gamspy._types import OperableType


def exp(x: OperableType) -> Expression:
    """
    Exponential of ``x`` (i.e. ``e^x``)

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import exp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = exp(a)
    >>> math.isclose(b.toValue(), 44.701184493300815)
    True

    """
    return expression.Expression(None, MathOp("exp", (x,)), None)


def log(x: OperableType) -> Expression:
    """
    Natural logarithm of ``x`` (i.e. logarithm base ``e`` of ``x``)

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log(a)
    >>> math.isclose(b.toValue(), 1.33500106673234)
    True

    """
    return expression.Expression(None, MathOp("log", (x,)), None)


def log_beta(x: OperableType, y: OperableType) -> Expression:
    """
    Log beta function (i.e. ``log(B(x, y)``)

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log_beta
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log_beta(a,5)
    >>> math.isclose(b.toValue(), -5.45446741772822)
    True

    """
    return expression.Expression(None, MathOp("logBeta", (x, y)), None)


def log_gamma(x: OperableType) -> Expression:
    """
    Log gamma function of ``x``

    Parameters
    ----------
    x : OperableType

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


def logit(x: OperableType) -> Expression:
    """
    Logit Transformation (i.e. ``log(x / (1 - x))``) for ``x`` in ``(0, 1)``

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import logit
    >>> m = Container()
    >>> a = Parameter(m, "a", records=0.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = logit(a)
    >>> math.isclose(b.toValue(), 1.3862943611198908)
    True

    """
    return expression.Expression(None, MathOp("logit", (x,)), None)


def log2(x: OperableType) -> Expression:
    """
    Binary logarithm (i.e. logarithm base ``2`` of ``x``)

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log2
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log2(a)
    >>> math.isclose(b.toValue(), 1.9259994185562224)
    True

    """
    return expression.Expression(None, MathOp("log2", (x,)), None)


def log10(x: OperableType) -> Expression:
    """
    Common logarithm (i.e. logarithm base ``10`` of ``x``)

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import log10
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = log10(a)
    >>> math.isclose(b.toValue(), 0.5797835966168101)
    True

    """
    return expression.Expression(None, MathOp("log10", (x,)), None)


def power(base: float | Operable, exponent: OperableType) -> Expression:
    """
    Base to the exponent power (i.e. ``base ^ exponent``)

    Parameters
    ----------
    base : OperableType
    exponent : OperableType

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import power
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = power(a, 3)
    >>> math.isclose(b.toValue(), 54.87199999999999)
    True

    """
    return expression.Expression(None, MathOp("power", (base, exponent)), None)


def cv_power(base: float, exponent: OperableType) -> Expression:
    """
    Real power (i.e. ``base ^ exponent`` where ``base >= 0``; error for ``base < 0``)

    Parameters
    ----------
    base : float
    exponent : OperableType

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


def rpower(base: OperableType | Operable, exponent: OperableType):
    """
    Returns ``x^y`` for ``x > 0`` and also for ``x = 0`` and restricted values of ``y`` (Error if ``x < 0``)

    Parameters
    ----------
    base : OperableType
    exponent : OperableType

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

    """
    return expression.Expression(
        None, MathOp("rPower", (base, exponent)), None
    )


def sign_power(base: OperableType, exponent: float):
    """
    Signed power: ``sign(base) * |base|^exponent``, for ``exponent > 0``

    Parameters
    ----------
    base : OperableType
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

    """
    if not isinstance(exponent, (float, int)):
        raise ValueError("Exponent must be a number")

    if exponent <= 0:
        raise ValueError("Exponent must be greater than 0")

    return expression.Expression(
        None, MathOp("signPower", (base, exponent)), None
    )


def sllog10(x: OperableType, S: int | float = 1.0e-150) -> Expression:
    """
    Smooth (linear) logarithm base 10

    Parameters
    ----------
    x : OperableType
    S : int | float, by default 1.0e-150

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sllog10
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = sllog10(a)
    >>> math.isclose(b.toValue(), 0.5797835966168101)
    True

    """
    return expression.Expression(None, MathOp("sllog10", (x, S)), None)


def sqlog10(x: OperableType, S: int | float = 1.0e-150) -> Expression:
    """
    Smooth (quadratic) logarithm base 10

    Parameters
    ----------
    x : OperableType
    S : int | float, by default 1.0e-150

    Returns
    -------
    Expression

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sqlog10
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = sqlog10(a)
    >>> math.isclose(b.toValue(), 0.5797835966168101)
    True

    """
    return expression.Expression(None, MathOp("sqlog10", (x, S)), None)


def vc_power(base: OperableType, exponent: OperableType):
    """
    Returns ``x^y`` for ``x >= 0`` (error for ``x < 0``)

    Parameters
    ----------
    base : OperableType
    exponent : OperableType

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


def sqr(x: OperableType) -> Expression:
    """
    Square of ``x`` (i.e. ``x^2``)

    Parameters
    ----------
    x : OperableType

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
    np.float64(16.0)

    """
    return expression.Expression(None, MathOp("sqr", (x,)), None)
