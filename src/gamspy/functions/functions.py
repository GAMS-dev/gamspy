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
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def beta(x: Union[int, float], y: Union[int, float]) -> "Expression":
    """
    Beta function

    Parameters
    ----------
    x : int | float
    y : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("beta(", f"{x},{y}", ")")


def regularized_beta(
    x: Union[int, float], y: Union[int, float], z: Union[int, float]
) -> "Expression":
    """
    Beta function

    Parameters
    ----------
    x : int | float
    y : int | float
    z : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("betaReg(", f"{x},{y},{z}", ")")


def gamma(x: Union[int, float], y: Union[int, float]) -> "Expression":
    """
    Gamma function

    Parameters
    ----------
    x : int | float
    y : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("gamma(", f"{x},{y}", ")")


def regularized_gamma(
    x: Union[int, float], y: Union[int, float], z: Union[int, float]
) -> "Expression":
    """
    Gamma function

    Parameters
    ----------
    x : int | float
    y : int | float
    z : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("gammaReg(", f"{x},{y},{z}", ")")


def lse_max(x: "Symbol") -> "Expression":
    """
    Smoothed Max via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression("lseMax(", x.gamsRepr(), ")")


def lse_max_sc(t: "Symbol", x: "Symbol") -> "Expression":
    """
    Scaled smoothed Max via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "lseMaxSc(", f"{t.gamsRepr()},{x.gamsRepr()}", ")"
    )


def lse_min(x: "Symbol") -> "Expression":
    """
    Smoothed Min via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression("lseMin(", x.gamsRepr(), ")")


def lse_min_sc(t: "Symbol", x: "Symbol") -> "Expression":
    """
    Scaled smoothed Min via the Logarithm of the Sum of Exponentials

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "lseMinSc(", f"{t.gamsRepr()},{x.gamsRepr()}", ")"
    )


def ncp_cm(
    x: "Symbol",
    y: "Symbol",
    z: Union[float, int],
) -> "Expression":
    """
    Chen-Mangasarian smoothing

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpCM(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
    )


def ncp_f(
    x: "Symbol",
    y: "Symbol",
    z: Union[int, float] = 0,
) -> "Expression":
    """
    Fisher-Burmeister smoothing

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpF(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
    )


def ncpVUpow(
    r: "Symbol",
    s: "Symbol",
    mu: Union[int, float] = 0,
) -> "Expression":
    """
    NCP Veelken-Ulbrich: smoothed min(r,s)

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpVUpow(", ",".join([r.gamsRepr(), s.gamsRepr(), str(mu)]), ")"
    )


def ncpVUsin(
    r: "Symbol",
    s: "Symbol",
    mu: Union[int, float] = 0,
) -> "Expression":
    """
    NCP Veelken-Ulbrich: smoothed min(r,s)

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "ncpVUsin(", ",".join([r.gamsRepr(), s.gamsRepr(), str(mu)]), ")"
    )


def poly(x: "Symbol", *args) -> "Expression":
    """
    Polynomial function

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    args_str = ",".join(str(arg) for arg in args)

    return expression.Expression("poly(", f"{x.gamsRepr()},{args_str}", ")")


def rand_binomial(n: Union[int, float], p: Union[int, float]) -> "Expression":
    """
    Generate a random number from the binomial distribution, where n is the
    number of trials and p the probability of success for each trial

    Parameters
    ----------
    n : int | float
    p : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("randBinomial(", f"{n},{p}", ")")


def rand_linear(
    low: Union[int, float], slope: Union[int, float], high: Union[int, float]
) -> "Expression":
    """
    Generate a random number between low and high with linear distribution.
    slope must be greater than 2 / (high - low)

    Parameters
    ----------
    low : int | float
    slope : int | float
    high : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("randLinear(", f"{low},{slope},{high}", ")")


def rand_triangle(
    low: Union[int, float], mid: Union[int, float], high: Union[int, float]
) -> "Expression":
    """
    Generate a random number between low and high with triangular distribution.
    mid is the most probable number.

    Parameters
    ----------
    low : int | float
    mid : int | float
    high : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression("randTriangle(", f"{low},{mid},{high}", ")")


def slrec(
    x: Union[int, float, "Symbol"], S: Union[int, float] = 1e-10
) -> "Expression":
    """
    Smooth (linear) reciprocal

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 1e-10

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("slrec(", f"{x_str},{S}", ")")


def sqrec(
    x: Union[int, float, "Symbol"], S: Union[int, float] = 1e-10
) -> "Expression":
    """
    Smooth (quadratic) reciprocal

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 1e-10

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("sqrec(", f"{x_str},{S}", ")")


def entropy(x: Union[int, float, "Symbol"]) -> "Expression":
    """
    L2 Norm of x

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("entropy(", x_str, ")")


def errorf(x: Union[int, float, "Symbol"]) -> "Expression":
    """
    Integral of the standard normal distribution

    Parameters
    ----------
    x : int, float, Symbol

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("errorf(", x_str, ")")
