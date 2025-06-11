from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.condition as condition
import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._validation as validation
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Alias, Set
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol
    from gamspy._symbols.symbol import Symbol
    from gamspy._types import OperableType


class MathOp(operable.Operable):
    def __init__(
        self,
        op_name: str,
        elements: tuple,
        safe_cancel: bool = False,
    ):
        self.op_name = op_name
        self.elements = list(elements)
        self.safe_cancel = safe_cancel
        self.container = None
        self.domain: list[Set | Alias] = []
        self.dimension = 0
        self.where = condition.Condition(self)
        if hasattr(elements[0], "container"):
            self.container = elements[0].container  # type: ignore
        if hasattr(elements[0], "domain"):
            self.domain: list[Set | Alias] = elements[0].domain  # type: ignore
            self.dimension = validation.get_dimension(self.domain)

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Evaluates the expression and returns the resulting records.

        Returns
        -------
        pd.DataFrame | None
        """
        assert self.container is not None
        temp_name = "a" + utils._get_unique_name()
        temp_param = syms.Parameter._constructor_bypass(
            self.container, temp_name, self.domain
        )
        temp_param[...] = self
        del self.container.data[temp_name]
        return temp_param.records

    def toValue(self) -> float | None:
        """
        Convenience method to return expression records as a Python float. Only possible if there is a single record as a result of the expression evaluation.

        Returns
        -------
        float | None

        Raises
        ------
        TypeError
            In case the dimension of the expression is not zero.
        """
        if self.dimension != 0:
            raise TypeError(
                f"Cannot extract value data for non-scalar expressions (expression dimension is {self.dimension})"
            )

        records = self.records
        if records is not None:
            return records["value"][0]

        return records

    def toList(self) -> list | None:
        """
        Convenience method to return the records of the expression as a list.

        Returns
        -------
        list | None
        """
        records = self.records
        if records is not None:
            return records.values.tolist()

        return None

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

    return x.gamsRepr()  # type: ignore


def abs(x: OperableType) -> MathOp:
    """
    Absolute value of ``x`` (i.e. ``|x|``)

    Returns
    -------
    MathOp

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
    return MathOp("abs", (x,))


def ceil(x: OperableType) -> MathOp:
    """
    The smallest integer greater than or equal to ``x`` (i.e. ``ceil(4.1)`` returns ``5``)

    Returns
    -------
    MathOp

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
    return MathOp("ceil", (x,))


def div(dividend: OperableType, divisor: OperableType) -> MathOp:
    """
    Dividing operation, Error if the divisor is ``0``. To avoid the error, ``div0`` can be used instead.

    Parameters
    ----------
    dividend : OperableType
    divisor : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("div", (dividend, divisor))


def div0(dividend: OperableType, divisor: OperableType) -> MathOp:
    """
    Dividing operation, returns ``1e+299`` if the divisor is ``0``

    Parameters
    ----------
    dividend : OperableType
    divisor : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("div0", (dividend, divisor))


def dist(
    x1: OperableType,
    x2: OperableType,
) -> MathOp:
    """
    Euclidean or L-2 Norm: ``sqrt(x1^2 + x2^2 + ... + xn^2)``

    Returns
    -------
    MathOp

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
    return MathOp("eDist", (x1, x2))


def factorial(x: int) -> MathOp:
    """
    Factorial of ``x``: ``x!``

    Parameters
    ----------
    x : int

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import factorial
    >>> m = Container()
    >>> b = Parameter(m, "b")
    >>> b[...] = factorial(2)

    """
    return MathOp("fact", (x,))


def floor(x: OperableType) -> MathOp:
    """
    The greatest integer less than or equal to ``x`` (i.e. ``floor(4.9)`` returns ``4``)

    Returns
    -------
    MathOp

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
    return MathOp("floor", (x,))


def fractional(x: OperableType) -> MathOp:
    """
    Returns the fractional part of ``x`` (i.e. ``fractional(3.9)`` returns ``0.9``)

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import fractional
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3.9)
    >>> b = Parameter(m, "b")
    >>> b[...] = fractional(a)
    >>> math.isclose(b.toValue(), 0.8999999999999999)
    True

    """
    return MathOp("frac", (x,))


def Min(*values) -> MathOp:
    """
    Minimum value of the values, where the number of values may vary.

    Returns
    -------
    MathOp

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
    return MathOp("min", values)


def Max(*values) -> MathOp:
    """
    Maximum value of the values, where the number of values may vary.

    Returns
    -------
    MathOp

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
    return MathOp("max", values)


def mod(x: OperableType, y: OperableType) -> MathOp:
    """
    Remainder of ``x`` divided by ``y`` (i.e. ``mod(10, 3)`` returns ``1``)

    Returns
    -------
    MathOp

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
    return MathOp("mod", (x, y))


def Round(x: OperableType, num_decimals: int = 0) -> MathOp:
    """
    Round ``x`` to ``num_decimals`` decimal places (i.e. ``Round(3.14159, 2)`` returns ``3.14``)

    Parameters
    ----------
    x : OperableType
    num_decimals : int, optional

    Returns
    -------
    MathOp

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
    return MathOp("round", (x, num_decimals))


def sign(x: Symbol) -> MathOp:
    """
    Sign of ``x`` returns ``1 if x > 0``, ``-1 if x < 0``, and ``0 if x = 0``

    Parameters
    ----------
    x : Symbol

    Returns
    -------
    MathOp

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
    return MathOp("sign", (x,))


def slexp(x: OperableType, S: int | float = 150) -> MathOp:
    """
    Smooth (linear) exponential where ``S <= 150``. (Default ``S = 150``)

    Parameters
    ----------
    x : OperableType
    S : int | float, by default 150

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import slexp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = slexp(a)
    >>> math.isclose(b.toValue(), 20.085536923187668)
    True

    """
    return MathOp("slexp", (x, S))


def sqexp(x: OperableType, S: int | float = 150) -> MathOp:
    """
    Smooth (quadratic) exponential where ``S <= 150``. (Default ``S = 150``)

    Parameters
    ----------
    x : OperableType
    S : int | float, by default 150

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sqexp
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = sqexp(a)
    >>> math.isclose(b.toValue(), 20.085536923187668)
    True

    """
    return MathOp("sqexp", (x, S))


def sqrt(x: OperableType, safe_cancel: bool = False) -> MathOp:
    """
    Square root of ``x``

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sqrt
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 54), ("i3", 0)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sqrt(a[i])

    """
    return MathOp("sqrt", (x,), safe_cancel=safe_cancel)


def truncate(x: OperableType) -> MathOp:
    """
    Returns the integer part of ``x`` (i.e. ``truncate(3.9)`` returns ``3``)

    Returns
    -------
    MathOp

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
    return MathOp("trunc", (x,))


def beta(x: OperableType, y: OperableType) -> MathOp:
    """
    Beta function: ``B(x, y) = gamma(x) * gamma(y) / gamma(x + y) = (x-1)! * (y-1)! / (x + y - 1)!``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import beta
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = beta(a, 1)
    >>> math.isclose(b.toValue(), 0.3333333333333333)
    True

    """
    return MathOp("beta", (x, y))


def regularized_beta(x: int | float, y: int | float, z: int | float) -> MathOp:
    """
    Regularized Beta Function, See `MathWorld <https://mathworld.wolfram.com/RegularizedBetaFunction.html>`_

    Parameters
    ----------
    x : int | float
    y : int | float
    z : int | float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import regularized_beta
    >>> m = Container()
    >>> a = Parameter(m, "a", records=3)
    >>> b = Parameter(m, "b")
    >>> b[...] = regularized_beta(0.5, a, 1)
    >>> math.isclose(b.toValue(), 0.12500000000000003)
    True

    """
    return MathOp("betaReg", (x, y, z))


def gamma(x: OperableType) -> MathOp:
    """
    Gamma function: ``gamma(x) = (x-1)!``

    Parameters
    ----------
    x : int | float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import gamma
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 7), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = gamma(a[i])

    """
    return MathOp("gamma", (x,))


def regularized_gamma(x: int | float, a: int | float) -> MathOp:
    """
    Lower Incomplete Regularized Gamma function, See `MathWorld <https://mathworld.wolfram.com/RegularizedGammaFunction.html>`_

    Parameters
    ----------
    x : int | float
    a : int | float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import regularized_gamma
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 1), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = regularized_gamma(0.5, a[i])

    """
    return MathOp("gammaReg", (x, a))


def lse_max(*xs) -> MathOp:
    """
    Smoothed Max via the Logarithm of the Sum of Exponentials: ``ln(exp(x1) + exp(x2) + ... + exp(xn))``

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_max
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 10), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_max(a[i], 5)

    """
    if len(xs) < 1:
        raise ValidationError("lse_max requires at least 1 x")

    return MathOp("lseMax", xs)


def lse_max_sc(t, *xs) -> MathOp:
    """
    Scaled smoothed Max via the Logarithm of the Sum of Exponentials: ``lse_max_sc(T,x) = lse_max(Tx)/T``

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_max_sc
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 100), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_max_sc(7.5, a[i], 10.5)

    """
    if len(xs) < 1:
        raise ValidationError("lse_max_sc requires at least 1 x")

    return MathOp("lseMaxSc", xs + (t,))


def lse_min(*xs) -> MathOp:
    """
    Smoothed Min via the Logarithm of the Sum of Exponentials: ``-ln(exp(-x1) + exp(-x2) + ... + exp(-xn))``

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import lse_min
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", 10), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = lse_min(a[i], 5)

    """
    if len(xs) < 1:
        raise ValidationError("lse_min requires at least 1 x")

    return MathOp("lseMin", xs)


def lse_min_sc(t, *xs) -> MathOp:
    """
    Scaled smoothed Min via the Logarithm of the Sum of Exponentials: ``lse_min_sc(T,x) = lse_min(Tx)/T``

    Returns
    -------
    MathOp

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

    return MathOp("lseMinSc", (t,) + xs)


def ncp_cm(x: Symbol, y: Symbol, z: float | int) -> MathOp:
    """
    Chen-Mangasarian smoothing: ``x - z*ln(1 + exp((x-y)/z))``

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ncp_cm
    >>> m = Container()
    >>> y = Parameter(m, "y", records=2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ncp_cm(1, y, 0.5)
    >>> math.isclose(b.toValue(), 0.9365359944785137)
    True

    """
    return MathOp("ncpCM", (x, y, z))


def ncp_f(x: Symbol, y: Symbol, z: int | float = 0) -> MathOp:
    """
    Fisher-Burmeister smoothing: ``sqrt(x^2 + y^2 + 2z) - x - y`` where ``z >= 0`` (default ``z = 0``)

    Parameters
    ----------
    x : Symbol
    y : Symbol
    z : int | float, optional

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import ncp_f
    >>> m = Container()
    >>> y = Parameter(m, "y", records=2)
    >>> b = Parameter(m, "b")
    >>> b[...] = ncp_f(1, y, 0.5)
    >>> math.isclose(b.toValue(), -0.5505102572168221)
    True

    """
    return MathOp("ncpF", (x, y, z))


def ncpVUpow(
    r: Symbol,
    s: Symbol,
    mu: int | float = 0,
) -> MathOp:
    """
    NCP Veelken-Ulbrich (smoothed min(r,s))

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    MathOp

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
    return MathOp("ncpVUpow", (r, s, mu))


def ncpVUsin(r: Symbol, s: Symbol, mu: int | float = 0) -> MathOp:
    """
    NCP Veelken-Ulbrich (smoothed min(r,s))

    Parameters
    ----------
    r : Symbol
    s : Symbol
    mu : int | float, optional

    Returns
    -------
    MathOp

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
    return MathOp("ncpVUsin", (r, s, mu))


def poly(x, *args) -> MathOp:
    """
    Polynomial function: ``p(x) = A[0] + A[1]*x + A[2]*x^2 + ... + A[n-1]*x^(n-1)``

    Returns
    -------
    MathOp

    Raises
    ------
    ValidationError
        If the number of arguments (args) is less than 3 or if any of args is not an integer or a float.

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

    return MathOp("poly", (x,) + args)


def sigmoid(x: OperableType) -> MathOp:
    """
    Sigmoid of ``x`` (i.e. ``1 / (1 + exp(-x))``)

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sigmoid
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 4), ("i2", -1), ("i3", 0.5)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sigmoid(a[i])

    """
    return MathOp("sigmoid", (x,))


def rand_binomial(n: int | float, p: int | float) -> MathOp:
    """
    Generate a random number from the binomial distribution, where n is the
    number of trials and p the probability of success for each trial

    Parameters
    ----------
    n : int | float
    p : int | float

    Returns
    -------
    MathOp

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
    return MathOp("randBinomial", (n, p))


def rand_linear(
    low: int | float, slope: int | float, high: int | float
) -> MathOp:
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
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import rand_linear
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> s = Parameter(m, "s", domain=i, records=[("i1", 0.03), ("i2", 0.008), ("i3", 0.04)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = rand_linear(75, s[i], 125)

    """
    return MathOp("randLinear", (low, slope, high))


def rand_triangle(
    low: int | float, mid: int | float, high: int | float
) -> MathOp:
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
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import rand_triangle
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> s = Parameter(m, "s", domain=i, records=[("i1", 103), ("i2", 80), ("i3", 115)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = rand_triangle(75, s[i], 125)

    """
    return MathOp("randTriangle", (low, mid, high))


def same_as(self: Set | Alias, other: Set | Alias | str) -> MathOp:
    """
    Evaluates to true if this set is identical to the given set or alias, false otherwise.

    Parameters
    ----------
    other : Set | Alias

    Returns
    -------
    MathOp

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
    return MathOp("sameAs", (self, other))


def slrec(x: OperableType, S: int | float = 1e-10) -> MathOp:
    """
    Smooth (linear) reciprocal, where ``S >= 1e-10``. (Default ``S = 1e-10``)

    Parameters
    ----------
    x : OperableType
    S : int | float, by default 1e-10

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import slrec
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 0.8), ("i3", 15)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = slrec(a[i])

    """
    return MathOp("slrec", (x, S))


def sqrec(x: OperableType, S: int | float = 1e-10) -> MathOp:
    """
    Smooth (quadratic) reciprocal, where ``S >= 1e-10``. (Default ``S = 1e-10``)

    Parameters
    ----------
    x : OperableType
    S : int | float, by default 1e-10

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import sqrec
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 0.8), ("i3", 15)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = sqrec(a[i])

    """
    return MathOp("sqrec", (x, S))


def entropy(x: OperableType) -> MathOp:
    """
    Entropy function: ``-x*ln(x)`` where ``x >= 0``

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import entropy
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", 1), ("i2", 0.8), ("i3", 15)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = entropy(a[i])

    """
    return MathOp("entropy", (x,))


def errorf(x: OperableType) -> MathOp:
    """
    Integral of the standard normal distribution from negative infinity to ``x``

    Parameters
    ----------
    x : int, float, Symbol

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import errorf
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> a = Parameter(m, "a", domain=i, records=[("i1", -2.5), ("i2", 0.8), ("i3", 1.7)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = errorf(a[i])

    """
    return MathOp("errorf", (x,))


def ifthen(
    condition: Expression,
    yes_return: float | Expression,
    no_return: float | Expression,
) -> MathOp:
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
    MathOp

    Examples
    --------
    >>> from gamspy.math import ifthen
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> tt = gp.Parameter(m, "tt", records=2)
    >>> y = gp.Parameter(m, "y", records=2)
    >>> x = ifthen(tt == 2, 3, 4 + y)

    """
    condition._representation = utils._replace_equality_signs(
        condition.gamsRepr()
    )

    return MathOp("ifthen", (condition, yes_return, no_return))


def bool_and(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff both ``x and y`` are true

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("bool_and", (x, y))


def bool_eqv(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``false`` iff exactly one argument is false

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("bool_eqv", (x, y))


def bool_imp(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x is false`` or ``y is true``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("bool_imp", (x, y))


def bool_not(x: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x is false``

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("bool_not", (x,))


def bool_or(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x is true`` or ``y is true``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("bool_or", (x, y))


def bool_xor(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff exactly one argument is ``false``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("bool_xor", (x, y))


def rel_eq(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x == y``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("rel_eq", (x, y))


def rel_ge(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x >= y``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("rel_ge", (x, y))


def rel_gt(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x > y``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("rel_gt", (x, y))


def rel_le(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x <= y``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("rel_le", (x, y))


def rel_lt(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x < y``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("rel_lt", (x, y))


def rel_ne(x: OperableType, y: OperableType) -> MathOp:
    """
    Returns ``true`` iff ``x != y``

    Parameters
    ----------
    x : OperableType
    y : OperableType

    Returns
    -------
    MathOp

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
    return MathOp("rel_ne", (x, y))


def map_value(x: OperableType) -> MathOp:
    """
    Returns an integer value that indicates what special value (if any) is stored in the input.
    Possible results:
    0: is not a special value
    4: is UNDF (undefined)
    5: is NA (not available)
    6: is INF
    7: is -INF
    8: is EPS

    Parameters
    ----------
    x : OperableType

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import map_value
    >>> m = Container()
    >>> a = Parameter(m, "a", records=12)
    >>> b = Parameter(m, "b")
    >>> b[...] = map_value(a)
    >>> b.toValue()
    np.float64(0.0)
    >>> a[...] = float('inf')
    >>> b[...] = map_value(a)
    >>> b.toValue()
    np.float64(6.0)

    """
    return MathOp("mapval", (x,))
