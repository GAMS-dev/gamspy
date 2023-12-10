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

from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import gamspy._algebra.expression as expression
import gamspy.utils as utils
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol
    from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol


def _stringify(x: Union[int, float, Symbol, ImplicitSymbol]):
    return str(x) if isinstance(x, (int, float)) else x.gamsRepr()


def abs(x: Union[int, float, Symbol]) -> Expression:
    """
    Absolute value of x (i.e. ``|x|``)

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"abs({x_str})", None)


def ceil(x: Union[int, float, Symbol]) -> Expression:
    """
    The smallest integer greater than or equal to x

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"ceil({x_str})", None)


def div(
    dividend: Union[int, float, Symbol], divisor: Union[int, float, Symbol]
) -> Expression:
    """
    Dividing operation

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression
    """
    dividend_str = _stringify(dividend)
    divisor_str = _stringify(divisor)

    return expression.Expression(
        None, f"div({dividend_str}, {divisor_str})", None
    )


def div0(
    dividend: Union[int, float, Symbol], divisor: Union[int, float, Symbol]
) -> Expression:
    """
    Dividing operation

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression
    """
    dividend_str = _stringify(dividend)
    divisor_str = _stringify(divisor)

    return expression.Expression(
        None, f"div0({dividend_str}, {divisor_str})", None
    )


def dist(
    x1: Union[Tuple[int, float], Symbol],
    x2: Union[Tuple[int, float], Symbol],
) -> Expression:
    """
    L2 norm

    Returns
    -------
    Expression

    Raises
    ------
    Exception
        In case both x1 and x2 are not a tuple or none.
    """
    if isinstance(x1, tuple) or isinstance(x2, tuple):
        raise Exception("Both should be a tuple or none")

    x1_str = _stringify(x1)
    x2_str = _stringify(x2)

    return expression.Expression(None, f"eDist({x1_str}, {x2_str})", None)


def factorial(x: Union[int, Symbol]) -> Expression:
    """
    Factorial of x

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"fact({x_str})", None)


def floor(x: Union[int, float, Symbol]) -> Expression:
    """
    The greatest integer less than or equal to x

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"floor({x_str})", None)


def fractional(x: Union[int, float, Symbol]) -> Expression:
    """
    Returns the fractional part of x

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"frac({x_str})", None)


def Min(*values) -> Expression:
    """
    Minimum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    values_str = ",".join([_stringify(value) for value in values])
    return expression.Expression(None, f"min({values_str})", None)


def Max(*values) -> Expression:
    """
    Maximum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    values_str = ",".join([_stringify(value) for value in values])
    return expression.Expression(None, f"max({values_str})", None)


def mod(x: Union[float, Symbol], y: Union[float, Symbol]) -> Expression:
    """
    Remainder of x divided by y.

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"mod({x_str},{y_str})", None)


def Round(x: Union[float, Symbol], num_decimals: int = 0) -> Expression:
    """
    Round x to num_decimals decimal places.

    Parameters
    ----------
    x : float | Symbol
    decimal : int, optional

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"round({x_str}, {num_decimals})", None)


def sign(x: Symbol) -> Expression:
    """
    Sign of x returns 1 if x > 0, -1 if x < 0, and 0 if x = 0

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"sign({x_str})", None)


def slexp(
    x: Union[int, float, Symbol], S: Union[int, float] = 150
) -> Expression:
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
    x_str = _stringify(x)
    return expression.Expression(None, f"slexp({x_str},{S})", None)


def sqexp(
    x: Union[int, float, Symbol], S: Union[int, float] = 150
) -> Expression:
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
    x_str = _stringify(x)
    return expression.Expression(None, f"sqexp({x_str},{S})", None)


def sqrt(x: Union[int, float, Symbol]) -> Expression:
    """
    Square root of x

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"sqrt({x_str})", None)


def truncate(x: Union[int, float, Symbol]) -> Expression:
    """
    Returns the integer part of x

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"trunc({x_str})", None)


def beta(
    x: Union[int, float, Symbol], y: Union[int, float, Symbol]
) -> Expression:
    """
    Beta function

    Parameters
    ----------
    x : int | float | Symbol
    y : int | float | Symbol

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"beta({x_str},{y_str})", None)


def regularized_beta(
    x: Union[int, float], y: Union[int, float], z: Union[int, float]
) -> Expression:
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
    x_str = _stringify(x)
    y_str = _stringify(y)
    z_str = _stringify(z)
    return expression.Expression(
        None, f"betaReg({x_str},{y_str},{z_str})", None
    )


def gamma(x: Union[int, float, Symbol]) -> Expression:
    """
    Gamma function

    Parameters
    ----------
    x : int | float

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"gamma({x_str})", None)


def regularized_gamma(
    x: Union[int, float], a: Union[int, float]
) -> Expression:
    """
    Gamma function

    Parameters
    ----------
    x : int | float
    a : int | float

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    a_str = _stringify(a)
    return expression.Expression(None, f"gammaReg({x_str},{a_str})", None)


def lse_max(*xs) -> Expression:
    """
    Smoothed Max via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    if len(xs) < 1:
        raise GamspyException("lse_max requires at least 1 x")

    x_str = ",".join([_stringify(x) for x in xs])
    return expression.Expression(None, f"lseMax({x_str})", None)


def lse_max_sc(t, *xs) -> Expression:
    """
    Scaled smoothed Max via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    t_str = _stringify(t)

    if len(xs) < 1:
        raise GamspyException("lse_max requires at least 1 x")

    x_str = ",".join([_stringify(x) for x in xs])

    return expression.Expression(None, f"lseMaxSc({t_str},{x_str})", None)


def lse_min(*xs) -> Expression:
    """
    Smoothed Min via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    if len(xs) < 1:
        raise GamspyException("lse_max requires at least 1 x")

    x_str = ",".join([_stringify(x) for x in xs])
    return expression.Expression(None, f"lseMin({x_str})", None)


def lse_min_sc(t, *xs) -> Expression:
    """
    Scaled smoothed Min via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    t_str = _stringify(t)

    if len(xs) < 1:
        raise GamspyException("lse_max requires at least 1 x")

    x_str = ",".join([_stringify(x) for x in xs])

    return expression.Expression(None, f"lseMinSc({t_str},{x_str})", None)


def ncp_cm(
    x: Symbol,
    y: Symbol,
    z: Union[float, int],
) -> Expression:
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
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"ncpCM({x_str},{y_str},{z})", None)


def ncp_f(
    x: Symbol,
    y: Symbol,
    z: Union[int, float] = 0,
) -> Expression:
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
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"ncpF({x_str},{y_str},{z})", None)


def ncpVUpow(
    r: Symbol,
    s: Symbol,
    mu: Union[int, float] = 0,
) -> Expression:
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
    r_str = _stringify(r)
    s_str = _stringify(s)
    return expression.Expression(None, f"ncpVUpow({r_str},{s_str},{mu}", None)


def ncpVUsin(
    r: Symbol,
    s: Symbol,
    mu: Union[int, float] = 0,
) -> Expression:
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
    r_str = _stringify(r)
    s_str = _stringify(s)
    return expression.Expression(None, f"ncpVUsin({r_str},{s_str},{mu})", None)


def poly(x, *args) -> Expression:
    """
    Polynomial function

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    args_str = ",".join(str(arg) for arg in args)

    return expression.Expression(None, f"poly({x_str},{args_str})", None)


def sigmoid(x: Union[int, float, Symbol]) -> Expression:
    """
    Sigmoid of x

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"sigmoid({x_str})", None)


def rand_binomial(n: Union[int, float], p: Union[int, float]) -> Expression:
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
    return expression.Expression(None, f"randBinomial({n},{p})", None)


def rand_linear(
    low: Union[int, float], slope: Union[int, float], high: Union[int, float]
) -> Expression:
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
    return expression.Expression(
        None, f"randLinear({low},{slope},{high})", None
    )


def rand_triangle(
    low: Union[int, float], mid: Union[int, float], high: Union[int, float]
) -> Expression:
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
    return expression.Expression(
        None, f"randTriangle({low},{mid},{high})", None
    )


def slrec(
    x: Union[int, float, Symbol], S: Union[int, float] = 1e-10
) -> Expression:
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
    x_str = _stringify(x)
    return expression.Expression(None, f"slrec({x_str},{S}", None)


def sqrec(
    x: Union[int, float, Symbol], S: Union[int, float] = 1e-10
) -> Expression:
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
    x_str = _stringify(x)
    return expression.Expression(None, f"sqrec({x_str},{S})", None)


def entropy(x: Union[int, float, Symbol]) -> Expression:
    """
    L2 Norm of x

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"entropy({x_str})", None)


def errorf(x: Union[int, float, Symbol]) -> Expression:
    """
    Integral of the standard normal distribution

    Parameters
    ----------
    x : int, float, Symbol

    Returns
    -------
    Expression
    """
    x_str = _stringify(x)
    return expression.Expression(None, f"errorf({x_str})", None)


def ifthen(
    condition: Expression,
    yes_return: Union[float, Expression],
    no_return: Union[float, Expression],
) -> Expression:
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
    >>> from gamspy.math import ifthen
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> tt = gp.Parameter(m, "tt", records=2)
    >>> y = gp.Parameter(m, "y", records=2)
    >>> x = ifthen(tt == 2, 3, 4 + y)

    """
    condition_str = condition.gamsRepr()
    condition_str = utils._replace_equality_signs(condition_str)

    ifthen_str = f"ifthen({condition_str}, {yes_return}, {no_return})"
    return expression.Expression(None, ifthen_str, None)


def bool_and(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"bool_and({x_str},{y_str})", None)


def bool_eqv(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"bool_eqv({x_str},{y_str})", None)


def bool_imp(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"bool_imp({x_str},{y_str})", None)


def bool_not(x: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    return expression.Expression(None, f"bool_not({x_str})", None)


def bool_or(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"bool_or({x_str},{y_str})", None)


def bool_xor(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"bool_xor({x_str},{y_str})", None)


def rel_eq(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"rel_eq({x_str},{y_str})", None)


def rel_ge(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"rel_ge({x_str},{y_str})", None)


def rel_gt(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"rel_gt({x_str},{y_str})", None)


def rel_le(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"rel_le({x_str},{y_str})", None)


def rel_lt(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"rel_lt({x_str},{y_str})", None)


def rel_ne(x: Union[int, Symbol], y: Union[int, Symbol]) -> Expression:
    x_str = _stringify(x)
    y_str = _stringify(y)
    return expression.Expression(None, f"rel_ne({x_str},{y_str})", None)
