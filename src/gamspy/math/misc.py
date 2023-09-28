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
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
import gamspy.utils as utils

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def abs(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Absolute value of x (i.e. ``|x|``)

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.fabs(x)
    return expression.Expression("abs(", x.gamsRepr(), ")")


def ceil(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    The smallest integer greater than or equal to x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.ceil(x)
    return expression.Expression("ceil(", x.gamsRepr(), ")")


def div(
    dividend: Union[int, float, "Symbol"], divisor: Union[int, float, "Symbol"]
) -> Union["Expression", float]:
    """
    Dividing operation

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression | float
    """
    if isinstance(dividend, (int, float)) and isinstance(
        divisor, (int, float)
    ):
        return dividend / divisor

    dividend_str = (
        str(dividend)
        if isinstance(dividend, (int, float))
        else dividend.gamsRepr()
    )
    divisor_str = (
        str(divisor)
        if isinstance(divisor, (int, float))
        else divisor.gamsRepr()
    )

    return expression.Expression("div(", f"{dividend_str}, {divisor_str}", ")")


def div0(
    dividend: Union[int, float, "Symbol"], divisor: Union[int, float, "Symbol"]
) -> Union["Expression", float]:
    """
    Dividing operation

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression | float
    """
    if isinstance(dividend, (int, float)) and isinstance(
        divisor, (int, float)
    ):
        if divisor == 0:
            return 1e299

        return dividend / divisor

    dividend_str = (
        str(dividend)
        if isinstance(dividend, (int, float))
        else dividend.gamsRepr()
    )
    divisor_str = (
        str(divisor)
        if isinstance(divisor, (int, float))
        else divisor.gamsRepr()
    )

    return expression.Expression(
        "div0(", f"{dividend_str}, {divisor_str}", ")"
    )


def dist(
    x1: Union[Tuple[int, float], "Symbol"],
    x2: Union[Tuple[int, float], "Symbol"],
) -> Union["Expression", float]:
    if isinstance(x1, tuple) and isinstance(x2, tuple):
        return math.dist(x1, x2)

    if isinstance(x1, tuple) or isinstance(x2, tuple):
        raise Exception("Both should be a tuple or none")

    x1_str = str(x1) if isinstance(x1, (int, float)) else x1.gamsRepr()
    x2_str = str(x2) if isinstance(x2, (int, float)) else x2.gamsRepr()

    return expression.Expression("eDist(", f"{x1_str}, {x2_str}", ")")


def factorial(x: Union[int, "Symbol"]) -> Union["Expression", int]:
    """
    Factorial of x

    Returns
    -------
    Expression | int
    """
    if isinstance(x, int):
        return math.factorial(x)
    return expression.Expression("fact(", x.gamsRepr(), ")")


def floor(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    The greatest integer less than or equal to x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.floor(x)
    return expression.Expression("floor(", x.gamsRepr(), ")")


def fractional(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Returns the fractional part of x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        fraction, _ = math.modf(x)
        return fraction
    return expression.Expression("frac(", x.gamsRepr(), ")")


def min(*values) -> "Expression":
    """
    Minimum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("min(", values_str, ")")


def max(*values) -> "Expression":
    """
    Maximum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    values_str = ",".join([value.gamsRepr() for value in values])
    return expression.Expression("max(", values_str, ")")


def mod(
    x: Union[float, "Symbol"], y: Union[float, "Symbol"]
) -> Union["Expression", int, float]:
    """
    Remainder of x divided by y.

    Returns
    -------
    Expression | int | float
    """
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x % y

    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    y_str = str(y) if isinstance(y, (int, float)) else y.gamsRepr()
    return expression.Expression("mod(" + x_str, ",", y_str + ")")


def Round(x: "Symbol", num_decimals: int = 0) -> "Expression":
    """
    Round x to num_decimals decimal places.

    Parameters
    ----------
    x : Operable
    decimal : int, optional

    Returns
    -------
    Expression
    """
    return expression.Expression(
        "round(", x.gamsRepr() + f", {num_decimals}", ")"
    )


def sign(x: "Symbol") -> "Expression":
    """
    Sign of x returns 1 if x > 0, -1 if x < 0, and 0 if x = 0

    Parameters
    ----------
    x : Operable

    Returns
    -------
    Expression
    """
    return expression.Expression("sign(", x.gamsRepr(), ")")


def slexp(
    x: Union[int, float, "Symbol"], S: Union[int, float] = 150
) -> "Expression":
    """
    Smooth (linear) exponential

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 150

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("slexp(", f"{x_str},{S}", ")")


def sqexp(
    x: Union[int, float, "Symbol"], S: Union[int, float] = 150
) -> "Expression":
    """
    Smooth (quadratic) exponential

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 150

    Returns
    -------
    Expression
    """
    x_str = str(x) if isinstance(x, (int, float)) else x.gamsRepr()
    return expression.Expression("sqexp(", f"{x_str},{S}", ")")


def sqrt(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Square root of x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        return math.sqrt(x)
    return expression.Expression("sqrt(", x.gamsRepr(), ")")


def truncate(x: Union[int, float, "Symbol"]) -> Union["Expression", float]:
    """
    Returns the integer part of x

    Returns
    -------
    Expression | float
    """
    if isinstance(x, (int, float)):
        _, integer = math.modf(x)
        return integer
    return expression.Expression("trunc(", x.gamsRepr(), ")")


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
