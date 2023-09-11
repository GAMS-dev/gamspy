#
# GAMS - General Algebraic Modeling System Python API
#
# Copyright (c) 2023 GAMS Development Corp. <support@gams.com>
# Copyright (c) 2023 GAMS Software GmbH <support@gams.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import math
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression

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
    base : float | Symbol
    exponent : float | Symbol

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
    base : float | Symbol
    exponent : float | Symbol

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


def rpower(base: Union[float, "Symbol"], exponent: Union[float, "Symbol"]):
    """
    Returns x^y for x > 0 and also for x = 0 and restricted values of y

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

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
    return expression.Expression("rPower(", f"{base_str},{exponent_str}", ")")


def sign_power(base: Union[float, "Symbol"], exponent: Union[float, "Symbol"]):
    """
    Signed power for y > 0.

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

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
    return expression.Expression(
        "signPower(", f"{base_str},{exponent_str}", ")"
    )


def sllog10(
    x: Union[int, float, "Symbol"], S: Union[int, float] = 1.0e-150
) -> "Expression":
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
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("sllog10(", f"{x_str},{S}", ")")


def sqlog10(
    x: Union[int, float, "Symbol"], S: Union[int, float] = 1.0e-150
) -> "Expression":
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
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("sqlog10(", f"{x_str},{S}", ")")


def vc_power(base: Union[float, "Symbol"], exponent: Union[float, "Symbol"]):
    """
    Returns x^y for x >= 0

    Parameters
    ----------
    base : float | Symbol
    exponent : float | Symbol

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
    return expression.Expression("vcPower(", f"{base_str},{exponent_str}", ")")


def sqr(x: Union[float, "Symbol"]) -> Union["Expression", float]:
    """
    Square of x

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression | float
    """
    return power(x, 2)
