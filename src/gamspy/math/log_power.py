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
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
from gamspy.math.misc import _stringify

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def exp(x: Union[float, Symbol]) -> Expression:
    """
    Exponential of x (i.e. e^x)

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"exp({x_str})", None)


def log(x: Union[int, float, Symbol]) -> Expression:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"log({x_str})", None)


def log_beta(
    x: Union[int, float, Symbol], y: Union[int, float, Symbol]
) -> Expression:
    """
    Log beta function

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"logBeta({x_str},{y_str})", None)


def log_gamma(
    x: Union[int, float, Symbol], y: Union[int, float, Symbol]
) -> Expression:
    """
    Log gamma function

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"logGamma({x_str},{y_str})", None)


def logit(x: Union[int, float, Symbol]) -> Expression:
    """
    Natural logarithm of x (i.e. logarithm base e of x)

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"logit({x_str})", None)


def log2(x: Union[float, Symbol]) -> Expression:
    """
    Binary logarithm (i.e. logarithm base 2 of x)

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"log2({x_str})", None)


def log10(x: Union[float, Symbol]) -> Expression:
    """
    Common logarithm (i.e. logarithm base 10 of x)

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"log10({x_str})", None)


def power(
    base: Union[float, Symbol], exponent: Union[float, Symbol]
) -> Expression:
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
    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression(
        None, f"power({base_str},{exponent_str})", None
    )


def cv_power(
    base: Union[float, Symbol], exponent: Union[float, Symbol]
) -> Expression:
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
    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression(
        None, f"cvPower({base_str},{exponent_str})", None
    )


def rpower(base: Union[float, Symbol], exponent: Union[float, Symbol]):
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
    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression(
        None, f"rPower({base_str},{exponent_str})", None
    )


def sign_power(base: Union[float, Symbol], exponent: Union[float, Symbol]):
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
    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression(
        None, f"signPower({base_str},{exponent_str})", None
    )


def sllog10(
    x: Union[int, float, Symbol], S: Union[int, float] = 1.0e-150
) -> Expression:
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
    x_str = _stringify(x)
    return expression.Expression(None, f"sllog10({x_str},{S})", None)


def sqlog10(
    x: Union[int, float, Symbol], S: Union[int, float] = 1.0e-150
) -> Expression:
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
    x_str = _stringify(x)
    return expression.Expression(None, f"sqlog10({x_str},{S})", None)


def vc_power(base: Union[float, Symbol], exponent: Union[float, Symbol]):
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
    base_str = (
        str(base) if isinstance(base, (int, float, str)) else base.gamsRepr()
    )
    exponent_str = (
        str(exponent)
        if isinstance(exponent, (int, float, str))
        else exponent.gamsRepr()
    )
    return expression.Expression(
        None, f"vcPower({base_str},{exponent_str})", None
    )


def sqr(x: Union[float, Symbol]) -> Expression:
    """
    Square of x

    Parameters
    ----------
    x : float | Symbol

    Returns
    -------
    Expression
    """
    return power(x, 2)
