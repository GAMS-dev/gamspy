from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy as gp
import gamspy._algebra.expression as expression
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Alias, Set
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
    from gamspy._symbols.symbol import Symbol


class MathOp:
    def __init__(
        self,
        op_name: str,
        elements: tuple,
    ):
        self.op_name = op_name
        self.elements = elements

    def gamsRepr(self):
        operands_str = ",".join([_stringify(elem) for elem in self.elements])
        return f"{self.op_name}({operands_str})"

    def _find_variables(self):
        variables = []
        for elem in self.elements:
            if isinstance(elem, gp.Variable):
                variables.append(elem.name)
            elif isinstance(elem, implicits.ImplicitVariable):
                variables.append(elem.parent.name)
            elif isinstance(elem, expression.Expression):
                variables += elem._find_variables()

        return variables

    def __str__(self):
        return self.gamsRepr()

    def __len__(self):
        return len(self.gamsRepr())


def _stringify(x: str | int | float | Symbol | ImplicitSymbol):
    if isinstance(x, (int, float)):
        x = utils._map_special_values(x)

        return str(x)
    elif isinstance(x, str):
        return f'"{x}"'

    return x.gamsRepr()


def abs(x: int | float | Symbol) -> Expression:
    """
    Absolute value of x (i.e. ``|x|``)

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("abs", (x,)), None)


def ceil(x: int | float | Symbol) -> Expression:
    """
    The smallest integer greater than or equal to x

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("ceil", (x,)), None)


def div(
    dividend: int | float | Symbol, divisor: int | float | Symbol
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
    return expression.Expression(
        None, MathOp("div", (dividend, divisor)), None
    )


def div0(
    dividend: int | float | Symbol, divisor: int | float | Symbol
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
    return expression.Expression(
        None, MathOp("div0", (dividend, divisor)), None
    )


def dist(
    x1: int | float | Symbol,
    x2: int | float | Symbol,
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
    return expression.Expression(None, MathOp("eDist", (x1, x2)), None)


def factorial(x: int | Symbol) -> Expression:
    """
    Factorial of x

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("fact", (x,)), None)


def floor(x: int | float | Symbol) -> Expression:
    """
    The greatest integer less than or equal to x

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("floor", (x,)), None)


def fractional(x: int | float | Symbol) -> Expression:
    """
    Returns the fractional part of x

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("frac", (x,)), None)


def Min(*values) -> Expression:
    """
    Minimum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("min", values), None)


def Max(*values) -> Expression:
    """
    Maximum value of the values, where the number of values may vary.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("max", values), None)


def mod(x: float | Symbol, y: float | Symbol) -> Expression:
    """
    Remainder of x divided by y.

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("mod", (x, y)), None)


def Round(x: float | Symbol, num_decimals: int = 0) -> Expression:
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
    return expression.Expression(
        None, MathOp("round", (x, num_decimals)), None
    )


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
    return expression.Expression(None, MathOp("sign", (x,)), None)


def slexp(x: int | float | Symbol, S: int | float = 150) -> Expression:
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
    return expression.Expression(None, MathOp("slexp", (x, S)), None)


def sqexp(x: int | float | Symbol, S: int | float = 150) -> Expression:
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
    return expression.Expression(None, MathOp("sqexp", (x, S)), None)


def sqrt(x: int | float | Symbol) -> Expression:
    """
    Square root of x

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sqrt", (x,)), None)


def truncate(x: int | float | Symbol) -> Expression:
    """
    Returns the integer part of x

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("trunc", (x,)), None)


def beta(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
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
    return expression.Expression(None, MathOp("beta", (x, y)), None)


def regularized_beta(
    x: int | float, y: int | float, z: int | float
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
    return expression.Expression(None, MathOp("betaReg", (x, y, z)), None)


def gamma(x: int | float | Symbol) -> Expression:
    """
    Gamma function

    Parameters
    ----------
    x : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("gamma", (x,)), None)


def regularized_gamma(x: int | float, a: int | float) -> Expression:
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
    return expression.Expression(None, MathOp("gammaReg", (x, a)), None)


def lse_max(*xs) -> Expression:
    """
    Smoothed Max via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    if len(xs) < 1:
        raise ValidationError("lse_max requires at least 1 x")

    return expression.Expression(None, MathOp("lseMax", xs), None)


def lse_max_sc(t, *xs) -> Expression:
    """
    Scaled smoothed Max via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    if len(xs) < 1:
        raise ValidationError("lse_max requires at least 1 x")

    return expression.Expression(None, MathOp("lseMaxSc", xs + (t,)), None)


def lse_min(*xs) -> Expression:
    """
    Smoothed Min via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    if len(xs) < 1:
        raise ValidationError("lse_max requires at least 1 x")

    return expression.Expression(None, MathOp("lseMin", xs), None)


def lse_min_sc(t, *xs) -> Expression:
    """
    Scaled smoothed Min via the Logarithm of the Sum of Exponentials

    Returns
    -------
    Expression
    """
    if len(xs) < 1:
        raise ValidationError("lse_max requires at least 1 x")

    return expression.Expression(None, MathOp("lseMinSc", (t,) + xs), None)


def ncp_cm(x: Symbol, y: Symbol, z: float | int) -> Expression:
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
    return expression.Expression(None, MathOp("ncpCM", (x, y, z)), None)


def ncp_f(x: Symbol, y: Symbol, z: int | float = 0) -> Expression:
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
    return expression.Expression(None, MathOp("ncpF", (x, y, z)), None)


def ncpVUpow(
    r: Symbol,
    s: Symbol,
    mu: int | float = 0,
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
    return expression.Expression(None, MathOp("ncpVUpow", (r, s, mu)), None)


def ncpVUsin(r: Symbol, s: Symbol, mu: int | float = 0) -> Expression:
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
    return expression.Expression(None, MathOp("ncpVUsin", (r, s, mu)), None)


def poly(x, *args) -> Expression:
    """
    Polynomial function

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("poly", (x,) + args), None)


def sigmoid(x: int | float | Symbol) -> Expression:
    """
    Sigmoid of x

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sigmoid", (x,)), None)


def rand_binomial(n: int | float, p: int | float) -> Expression:
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
    return expression.Expression(None, MathOp("randBinomial", (n, p)), None)


def rand_linear(
    low: int | float, slope: int | float, high: int | float
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
        None, MathOp("randLinear", (low, slope, high)), None
    )


def rand_triangle(
    low: int | float, mid: int | float, high: int | float
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
        None, MathOp("randTriangle", (low, mid, high)), None
    )


def same_as(self: Set | Alias, other: Set | Alias | str) -> Expression:
    """
    Evaluates to true if this set is identical to the given set or alias, false otherwise.

    Parameters
    ----------
    other : Set | Alias

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("sameAs", (self, other)), None)


def slrec(x: int | float | Symbol, S: int | float = 1e-10) -> Expression:
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
    return expression.Expression(None, MathOp("slrec", (x, S)), None)


def sqrec(x: int | float | Symbol, S: int | float = 1e-10) -> Expression:
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
    return expression.Expression(None, MathOp("sqrec", (x, S)), None)


def entropy(x: int | float | Symbol) -> Expression:
    """
    L2 Norm of x

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("entropy", (x,)), None)


def errorf(x: int | float | Symbol) -> Expression:
    """
    Integral of the standard normal distribution

    Parameters
    ----------
    x : int, float, Symbol

    Returns
    -------
    Expression
    """
    return expression.Expression(None, MathOp("errorf", (x,)), None)


def ifthen(
    condition: Expression,
    yes_return: float | Expression,
    no_return: float | Expression,
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
    condition.representation = utils._replace_equality_signs(
        condition.gamsRepr()
    )

    return expression.Expression(
        None, MathOp("ifthen", (condition, yes_return, no_return)), None
    )


def bool_and(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("bool_and", (x, y)), None)


def bool_eqv(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("bool_eqv", (x, y)), None)


def bool_imp(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("bool_imp", (x, y)), None)


def bool_not(x: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("bool_not", (x,)), None)


def bool_or(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("bool_or", (x, y)), None)


def bool_xor(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("bool_xor", (x, y)), None)


def rel_eq(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("rel_eq", (x, y)), None)


def rel_ge(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("rel_ge", (x, y)), None)


def rel_gt(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("rel_gt", (x, y)), None)


def rel_le(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("rel_le", (x, y)), None)


def rel_lt(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("rel_lt", (x, y)), None)


def rel_ne(x: int | Symbol, y: int | Symbol) -> Expression:
    return expression.Expression(None, MathOp("rel_ne", (x, y)), None)
