from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
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
        safe_cancel: bool = False,
    ):
        self.op_name = op_name
        self.elements = elements
        self.safe_cancel = safe_cancel

    def gamsRepr(self) -> str:
        operands_str = ",".join([_stringify(elem) for elem in self.elements])
        return f"{self.op_name}({operands_str})"

    def latexRepr(self) -> str:
        """
        Representation of this MathOp in Latex.

        Returns
        -------
        str
        """
        op_map = {
            "sqrt": "\\sqrt",
            "floor": "\\floor",
            "ceil": "\\lceil",
            "abs": "\\lvert",
        }

        operands_str = ",".join([_stringify(elem) for elem in self.elements])
        if self.op_name in op_map:
            return f"{op_map[self.op_name]}{{{operands_str}}}"

        return f"{self.op_name}({operands_str})"

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
    Absolute value of ``x`` (i.e. ``|x|``)

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import abs
    >>> m = Container()
    >>> a = Parameter(m, "a", records=-3.8)
    >>> b = Parameter(m, "b")
    >>> b[...] = abs(a)
    >>> b.toValue()
    np.float64(3.8)

    """
    return expression.Expression(None, MathOp("abs", (x,)), None)


def ceil(x: int | float | Symbol) -> Expression:
    """
    The smallest integer greater than or equal to ``x`` (i.e. ``ceil(4.1)`` returns ``5``)

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ceil
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ceil(a)
    >>> b.toValue()
    np.float64(4.0)

    """
    return expression.Expression(None, MathOp("ceil", (x,)), None)


def div(
    dividend: int | float | Symbol, divisor: int | float | Symbol
) -> Expression:
    """
    Dividing operation, Error if the divisor is ``0``. To avoid the error, ``div0`` can be used instead.

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import div
    >>> m = Container()
    >>> a = Parameter(m, "a", records=210)
    >>> b = Parameter(m, "b")
    >>> b[...] = div(a, 3)
    >>> b.toValue()
    np.float64(70.0)

    """
    return expression.Expression(
        None, MathOp("div", (dividend, divisor)), None
    )


def div0(
    dividend: int | float | Symbol, divisor: int | float | Symbol
) -> Expression:
    """
    Dividing operation, returns ``1e+299`` if the divisor is ``0``

    Parameters
    ----------
    dividend : int | float | Symbol
    divisor : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import div0
    >>> m = Container()
    >>> a = Parameter(m, "a", records=210)
    >>> b = Parameter(m, "b")
    >>> b[...] = div0(a, 0)
    >>> b.toValue()
    np.float64(1e+299)

    """
    return expression.Expression(
        None, MathOp("div0", (dividend, divisor)), None
    )


def dist(
    x1: int | float | Symbol,
    x2: int | float | Symbol,
) -> Expression:
    """
    Euclidean or L-2 Norm: ``sqrt(x1^2 + x2^2 + ... + xn^2)``

    Returns
    -------
    Expression

    Raises
    ------
    Exception
        In case both x1 and x2 are not a tuple or none.

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import dist
    >>> m = Container()
    >>> a = Parameter(m, "a", records=210)
    >>> b = Parameter(m, "b")
    >>> b[...] = dist(a, 100)

    """
    return expression.Expression(None, MathOp("eDist", (x1, x2)), None)


def factorial(x: int) -> Expression:
    """
    Factorial of ``x``: ``x!``

    Parameters
    ----------
    x : int

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import factorial
    >>> m = Container()
    >>> b = Parameter(m, "b")
    >>> b[...] = factorial(2)

    """
    if not isinstance(x, int):
        raise ValidationError("Factorial requires an integer")

    return expression.Expression(None, MathOp("fact", (x,)), None)


def floor(x: int | float | Symbol) -> Expression:
    """
    The greatest integer less than or equal to ``x`` (i.e. ``floor(4.9)`` returns ``4``)

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import floor
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.9)
    >>> b = Parameter(m, "b")
    >>> b[...] = floor(a)
    >>> b.toValue()
    np.float64(3.0)

    """
    return expression.Expression(None, MathOp("floor", (x,)), None)


def fractional(x: int | float | Symbol) -> Expression:
    """
    Returns the fractional part of ``x`` (i.e. ``fractional(3.9)`` returns ``0.9``)

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import fractional
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.9)
    >>> b = Parameter(m, "b")
    >>> b[...] = fractional(a)
    >>> b.toValue()
    np.float64(0.8999999999999999)

    """
    return expression.Expression(None, MathOp("frac", (x,)), None)


def Min(*values) -> Expression:
    """
    Minimum value of the values, where the number of values may vary.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import Min
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", -2), ("i2", 0.3), ("i3", 2)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = Min(a[i], 1)
    >>> b.toList()
    [('i1', -2.0), ('i2', 0.3), ('i3', 1.0)]

    """
    return expression.Expression(None, MathOp("min", values), None)


def Max(*values) -> Expression:
    """
    Maximum value of the values, where the number of values may vary.

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import Max
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 2), ("i2", 0.3), ("i3", 2.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = Max(a[i], 1)
    >>> b.toList()
    [('i1', 2.0), ('i2', 1.0), ('i3', 2.5)]

    """
    return expression.Expression(None, MathOp("max", values), None)


def mod(x: float | Symbol, y: float | Symbol) -> Expression:
    """
    Remainder of ``x`` divided by ``y`` (i.e. ``mod(10, 3)`` returns ``1``)

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import mod
    >>> m = Container()
    >>> a = Parameter(m, "a", records=200)
    >>> b = Parameter(m, "b")
    >>> b[...] = mod(a, 3)
    >>> b.toValue()
    np.float64(2.0)

    """
    return expression.Expression(None, MathOp("mod", (x, y)), None)


def Round(x: float | Symbol, num_decimals: int = 0) -> Expression:
    """
    Round ``x`` to ``num_decimals`` decimal places (i.e. ``Round(3.14159, 2)`` returns ``3.14``)

    Parameters
    ----------
    x : float | Symbol
    num_decimals : int, optional

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import Round, div
    >>> m = Container()
    >>> a = Parameter(m, "a", records=200)
    >>> b = Parameter(m, "b")
    >>> b[...] = Round(div(a, 3), 2)
    >>> b.toValue()
    np.float64(66.67)

    """
    if not isinstance(num_decimals, int):
        raise ValidationError("Round requires num_decimals to be an integer")

    return expression.Expression(
        None, MathOp("round", (x, num_decimals)), None
    )


def sign(x: Symbol) -> Expression:
    """
    Sign of ``x`` returns ``1 if x > 0``, ``-1 if x < 0``, and ``0 if x = 0``

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sign
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 2), ("i2", -5.4), ("i3", 0)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sign(a[i])
    >>> b.toList()
    [('i1', 1.0), ('i2', -1.0)]

    """
    return expression.Expression(None, MathOp("sign", (x,)), None)


def slexp(x: int | float | Symbol, S: int | float = 150) -> Expression:
    """
    Smooth (linear) exponential where ``S <= 150``. (Default ``S = 150``)

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 150

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import slexp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = slexp(a)
    >>> b.toValue()
    np.float64(20.085536923187668)

    """
    return expression.Expression(None, MathOp("slexp", (x, S)), None)


def sqexp(x: int | float | Symbol, S: int | float = 150) -> Expression:
    """
    Smooth (quadratic) exponential where ``S <= 150``. (Default ``S = 150``)

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 150

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sqexp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = sqexp(a)
    >>> b.toValue()
    np.float64(20.085536923187668)

    """
    return expression.Expression(None, MathOp("sqexp", (x, S)), None)


def sqrt(x: int | float | Symbol, safe_cancel: bool = False) -> Expression:
    """
    Square root of ``x``

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sqrt
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 54), ("i3", 0)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sqrt(a[i])
    >>> b.toList()
    [('i1', 2.0), ('i2', 7.3484692283495345)]

    """
    return expression.Expression(
        None, MathOp("sqrt", (x,), safe_cancel=safe_cancel), None
    )


def truncate(x: int | float | Symbol) -> Expression:
    """
    Returns the integer part of ``x`` (i.e. ``truncate(3.9)`` returns ``3``)

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import truncate
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.9)
    >>> b = Parameter(m, "b")
    >>> b[...] = truncate(a)
    >>> b.toValue()
    np.float64(3.0)

    """
    return expression.Expression(None, MathOp("trunc", (x,)), None)


def beta(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Beta function: ``B(x, y) = gamma(x) * gamma(y) / gamma(x + y) = (x-1)! * (y-1)! / (x + y - 1)!``

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
    >>> from gamspy.math import beta
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = beta(a, 1)
    >>> b.toValue()
    np.float64(0.3333333333333333)

    """
    return expression.Expression(None, MathOp("beta", (x, y)), None)


def regularized_beta(
    x: int | float, y: int | float, z: int | float
) -> Expression:
    """
    Regularized Beta Function, See `MathWorld <https://mathworld.wolfram.com/RegularizedBetaFunction.html>`_

    Parameters
    ----------
    x : int | float
    y : int | float
    z : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import regularized_beta
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = regularized_beta(0.5, a, 1)
    >>> b.toValue()
    np.float64(0.12500000000000003)

    """
    return expression.Expression(None, MathOp("betaReg", (x, y, z)), None)


def gamma(x: int | float | Symbol) -> Expression:
    """
    Gamma function: ``gamma(x) = (x-1)!``

    Parameters
    ----------
    x : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import gamma
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 7), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = gamma(a[i])
    >>> b.toList()
    [('i1', 6.0), ('i2', 720.0), ('i3', 1.772453850905516)]

    """
    return expression.Expression(None, MathOp("gamma", (x,)), None)


def regularized_gamma(x: int | float, a: int | float) -> Expression:
    """
    Lower Incomplete Regularized Gamma function, See `MathWorld <https://mathworld.wolfram.com/RegularizedGammaFunction.html>`_

    Parameters
    ----------
    x : int | float
    a : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import regularized_gamma
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 1), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = regularized_gamma(0.5, a[i])
    >>> b.toList()
    [('i1', 0.001751622556290824), ('i2', 0.3934693402873665), ('i3', 0.6826894921370857)]

    """
    return expression.Expression(None, MathOp("gammaReg", (x, a)), None)


def lse_max(*xs) -> Expression:
    """
    Smoothed Max via the Logarithm of the Sum of Exponentials: ``ln(exp(x1) + exp(x2) + ... + exp(xn))``

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_max
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 10), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_max(a[i], 5)
    >>> b.toList()
    [('i1', 5.313261687518223), ('i2', 10.006715348489118), ('i3', 5.011047744848594)]

    """
    if len(xs) < 1:
        raise ValidationError("lse_max requires at least 1 x")

    return expression.Expression(None, MathOp("lseMax", xs), None)


def lse_max_sc(t, *xs) -> Expression:
    """
    Scaled smoothed Max via the Logarithm of the Sum of Exponentials: ``lse_max_sc(T,x) = lse_max(Tx)/T``

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_max_sc
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 100), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_max_sc(7.5, a[i], 10.5)
    >>> b.toList()
    [('i1', 10.50000153604837), ('i2', 10.5), ('i3', 10.902826555965506)]

    """
    if len(xs) < 1:
        raise ValidationError("lse_max_sc requires at least 1 x")

    return expression.Expression(None, MathOp("lseMaxSc", xs + (t,)), None)


def lse_min(*xs) -> Expression:
    """
    Smoothed Min via the Logarithm of the Sum of Exponentials: ``-ln(exp(-x1) + exp(-x2) + ... + exp(-xn))``

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_min
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 10), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_min(a[i], 5)
    >>> b.toList()
    [('i1', 3.686738312481777), ('i2', 4.993284651510882), ('i3', 0.4889522551514062)]

    """
    if len(xs) < 1:
        raise ValidationError("lse_min requires at least 1 x")

    return expression.Expression(None, MathOp("lseMin", xs), None)


def lse_min_sc(t, *xs) -> Expression:
    """
    Scaled smoothed Min via the Logarithm of the Sum of Exponentials: ``lse_min_sc(T,x) = lse_min(Tx)/T``

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_min_sc
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 100), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_min_sc(7.5, a[i], 10.5)
    >>> b.toList()
    [('i1', 4.0), ('i2', 10.5), ('i3', 0.5)]

    """
    if len(xs) < 1:
        raise ValidationError("lse_min_sc requires at least 1 x")

    return expression.Expression(None, MathOp("lseMinSc", (t,) + xs), None)


def ncp_cm(x: Symbol, y: Symbol, z: float | int) -> Expression:
    """
    Chen-Mangasarian smoothing: ``x - z*ln(1 + exp((x-y)/z))``

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ncp_cm
    >>> m = Container()
    >>> y = Parameter(m, "y", records=2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ncp_cm(1, y, 0.5)
    >>> b.toValue()
    np.float64(0.9365359944785137)

    """
    if not isinstance(z, (int, float)):
        raise ValidationError("ncp_cm requires z to be an integer or a float")

    if z <= 0:
        raise ValidationError("ncp_cm requires z to be greater than 0")

    return expression.Expression(None, MathOp("ncpCM", (x, y, z)), None)


def ncp_f(x: Symbol, y: Symbol, z: int | float = 0) -> Expression:
    """
    Fisher-Burmeister smoothing: ``sqrt(x^2 + y^2 + 2z) - x - y`` where ``z >= 0`` (default ``z = 0``)

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float, optional

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ncp_f
    >>> m = Container()
    >>> y = Parameter(m, "y", records=2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ncp_f(1, y, 0.5)
    >>> b.toValue()
    np.float64(-0.5505102572168221)

    """
    if not isinstance(z, (int, float)):
        raise ValidationError("ncp_f requires z to be an integer or a float")

    if z < 0:
        raise ValidationError(
            "ncp_f requires z to be greater than or equal to 0"
        )

    return expression.Expression(None, MathOp("ncpF", (x, y, z)), None)


def ncpVUpow(
    r: Symbol,
    s: Symbol,
    mu: int | float = 0,
) -> Expression:
    """
    NCP Veelken-Ulbrich (smoothed min(r,s))

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ncpVUpow
    >>> m = Container()
    >>> y = Parameter(m, "y", records=2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ncpVUpow(1, y, 0.5)
    >>> b.toValue()
    np.float64(1.0)

    """
    if not isinstance(mu, (int, float)):
        raise ValidationError(
            "ncpVUpow requires mu to be an integer or a float"
        )

    return expression.Expression(None, MathOp("ncpVUpow", (r, s, mu)), None)


def ncpVUsin(r: Symbol, s: Symbol, mu: int | float = 0) -> Expression:
    """
    NCP Veelken-Ulbrich (smoothed min(r,s))

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ncpVUsin
    >>> m = Container()
    >>> y = Parameter(m, "y", records=2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ncpVUsin(1, y, 0.5)
    >>> b.toValue()
    np.float64(1.0)

    """
    if not isinstance(mu, (int, float)):
        raise ValidationError(
            "ncpVUsin requires mu to be an integer or a float"
        )

    return expression.Expression(None, MathOp("ncpVUsin", (r, s, mu)), None)


def poly(x, *args) -> Expression:
    """
    Polynomial function: ``p(x) = A[0] + A[1]*x + A[2]*x^2 + ... + A[n-1]*x^(n-1)``

    Returns
    -------
    Expression

    Exceptions
    ----------
    ValidationError: If the number of arguments (args) is less than 3 or if any of args is not an integer or a float.

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import poly
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 10), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = poly(a[i], 15, 3, 4)
    >>> b.toList()
    [('i1', 91.0), ('i2', 445.0), ('i3', 17.5)]

    """
    if len(args) < 3:
        raise ValidationError("poly requires at least 3 arguments after x")

    if not all(isinstance(arg, (int, float)) for arg in args):
        raise ValidationError(
            "poly requires all args to be integers or floats"
        )

    return expression.Expression(None, MathOp("poly", (x,) + args), None)


def sigmoid(x: int | float | Symbol) -> Expression:
    """
    Sigmoid of ``x`` (i.e. ``1 / (1 + exp(-x))``)

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sigmoid
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", -1), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sigmoid(a[i])
    >>> b.toList()
    [('i1', 0.9820137900379085), ('i2', 0.2689414213699951), ('i3', 0.6224593312018546)]

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

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import rand_binomial
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> p = Parameter(m, "p", domain=i, records=[("i1", 0.3), ("i2", 0.8), ("i3", 0.45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = rand_binomial(75, p[i])
    >>> b.toList()
    [('i1', 21.0), ('i2', 63.0), ('i3', 25.0)]

    """
    return expression.Expression(None, MathOp("randBinomial", (n, p)), None)


def rand_linear(
    low: int | float, slope: int | float, high: int | float
) -> Expression:
    """
    Generate a random number between low and high with linear distribution.
    ``slope`` must be less than ``2 / (high - low)`` and greater than ``0``

    Parameters
    ----------
    low : int | float
    slope : int | float
    high : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import rand_linear
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> s = Parameter(m, "s", domain=i, records=[("i1", 0.03), ("i2", 0.008), ("i3", 0.04)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = rand_linear(75, s[i], 125)
    >>> b.toList()
    [('i1', 78.22119203430918), ('i2', 87.65662570307367), ('i3', 80.24583337516547)]

    """
    return expression.Expression(
        None, MathOp("randLinear", (low, slope, high)), None
    )


def rand_triangle(
    low: int | float, mid: int | float, high: int | float
) -> Expression:
    """
    Generate a random number between ``low`` and ``high`` with triangular distribution.
    ``mid`` is the most probable number.

    Parameters
    ----------
    low : int | float
    mid : int | float
    high : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import rand_triangle
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> s = Parameter(m, "s", domain=i, records=[("i1", 103), ("i2", 80), ("i3", 115)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = rand_triangle(75, s[i], 125)
    >>> b.toList()
    [('i1', 90.50632080153123), ('i2', 106.22102486822031), ('i3', 108.17756338250294)]

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

    Examples
    --------
    >>> import gamspy as gp
    >>> from gamspy.math import same_as
    >>> m = gp.Container()
    >>> i = gp.Set(m, name="i", records=["seattle", "san-diego"])
    >>> j = gp.Set(m, name="j", records=["new-york", "seattle"])
    >>> attr = gp.Parameter(m, "attr", domain = [i, j])
    >>> attr[i,j]  =  same_as(i, j)
    >>> attr.records.values.tolist()
    [['seattle', 'seattle', 1.0]]

    """
    return expression.Expression(None, MathOp("sameAs", (self, other)), None)


def slrec(x: int | float | Symbol, S: int | float = 1e-10) -> Expression:
    """
    Smooth (linear) reciprocal, where ``S >= 1e-10``. (Default ``S = 1e-10``)

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 1e-10

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import slrec
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 0.8), ("i3", 15)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = slrec(a[i])
    >>> b.toList()
    [('i1', 1.0), ('i2', 1.25), ('i3', 0.06666666666666667)]

    """
    return expression.Expression(None, MathOp("slrec", (x, S)), None)


def sqrec(x: int | float | Symbol, S: int | float = 1e-10) -> Expression:
    """
    Smooth (quadratic) reciprocal, where ``S >= 1e-10``. (Default ``S = 1e-10``)

    Parameters
    ----------
    x : int | float | Symbol
    S : int | float, by default 1e-10

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sqrec
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 0.8), ("i3", 15)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sqrec(a[i])
    >>> b.toList()
    [('i1', 1.0), ('i2', 1.25), ('i3', 0.06666666666666667)]

    """
    return expression.Expression(None, MathOp("sqrec", (x, S)), None)


def entropy(x: int | float | Symbol) -> Expression:
    """
    Entropy function: ``-x*ln(x)`` where ``x >= 0``

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import entropy
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 0.8), ("i3", 15)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = entropy(a[i])
    >>> b.toList()
    [('i2', 0.17851484105136778), ('i3', -40.62075301653315)]

    """
    return expression.Expression(None, MathOp("entropy", (x,)), None)


def errorf(x: int | float | Symbol) -> Expression:
    """
    Integral of the standard normal distribution from negative infinity to ``x``

    Parameters
    ----------
    x : int, float, Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import errorf
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", -2.5), ("i2", 0.8), ("i3", 1.7)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = errorf(a[i])
    >>> b.toList()
    [('i1', 0.0062096653257761375), ('i2', 0.7881446014166034), ('i3', 0.955434537241457)]

    """
    return expression.Expression(None, MathOp("errorf", (x,)), None)


def ifthen(
    condition: Expression,
    yes_return: float | Expression,
    no_return: float | Expression,
) -> Expression:
    """
    If the logical condition is ``true``, the function returns ``yes_return``,
    else it returns ``no_return``

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


def bool_and(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff both ``x and y`` are true

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
    >>> from gamspy.math import bool_and
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = bool_and(a > 10, b < 5)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("bool_and", (x, y)), None)


def bool_eqv(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``false`` iff exactly one argument is false

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
    >>> from gamspy.math import bool_eqv
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = bool_eqv(a > 10, b < 5)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("bool_eqv", (x, y)), None)


def bool_imp(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x is false`` or ``y is true``

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
    >>> from gamspy.math import bool_imp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = bool_imp(a < 10, b > 5)
    >>> c.toValue()
    np.float64(1.0)

    """
    return expression.Expression(None, MathOp("bool_imp", (x, y)), None)


def bool_not(x: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x is false``

    Parameters
    ----------
    x : int | float | Symbol

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import bool_not
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = bool_not(a > 10)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("bool_not", (x,)), None)


def bool_or(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x is true`` or ``y is true``

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
    >>> from gamspy.math import bool_or
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = bool_or(a > 15, b < 5)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("bool_or", (x, y)), None)


def bool_xor(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff exactly one argument is ``false``

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
    >>> from gamspy.math import bool_xor
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = bool_xor(a < 15, b > 5)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("bool_xor", (x, y)), None)


def rel_eq(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x == y``

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
    >>> from gamspy.math import rel_eq
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = rel_eq(a, b)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("rel_eq", (x, y)), None)


def rel_ge(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x >= y``

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
    >>> from gamspy.math import rel_ge
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = rel_ge(a, b)
    >>> c.toValue()
    np.float64(1.0)

    """
    return expression.Expression(None, MathOp("rel_ge", (x, y)), None)


def rel_gt(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x > y``

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
    >>> from gamspy.math import rel_gt
    >>> m = Container()
    >>> a = Parameter(m, "a", records=7)
    >>> b = Parameter(m, "b", records=7)
    >>> c = Parameter(m, "c")
    >>> c[...] = rel_gt(a, b)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("rel_gt", (x, y)), None)


def rel_le(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x <= y``

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
    >>> from gamspy.math import rel_le
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=11)
    >>> c = Parameter(m, "c")
    >>> c[...] = rel_le(a, b)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("rel_le", (x, y)), None)


def rel_lt(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x < y``

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
    >>> from gamspy.math import rel_lt
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=17)
    >>> c = Parameter(m, "c")
    >>> c[...] = rel_lt(a, b)
    >>> c.toValue()
    np.float64(1.0)

    """
    return expression.Expression(None, MathOp("rel_lt", (x, y)), None)


def rel_ne(x: int | float | Symbol, y: int | float | Symbol) -> Expression:
    """
    Returns ``true`` iff ``x != y``

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
    >>> from gamspy.math import rel_ne
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b", records=12)
    >>> c = Parameter(m, "c")
    >>> c[...] = rel_ne(a, b)
    >>> c.toValue()
    np.float64(0.0)

    """
    return expression.Expression(None, MathOp("rel_ne", (x, y)), None)
