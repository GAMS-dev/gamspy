from __future__ import annotations

from typing import TYPE_CHECKING

from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._types import OperableType


def cos(x: OperableType) -> MathOp:
    """
    Cosine of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import cos
    >>> import numpy as np
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = cos(np.pi)
    >>> r.toValue()
    np.float64(-1.0)

    """
    return MathOp("cos", (x,))


def cosh(x: OperableType) -> MathOp:
    """
    Hyperbolic cosine of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import cosh
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = cosh(0)
    >>> r.toValue()
    np.float64(1.0)

    """
    return MathOp("cosh", (x,))


def sin(x: OperableType) -> MathOp:
    """
    Sine of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sin
    >>> import numpy as np
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = sin(np.pi/2)
    >>> r.toValue()
    np.float64(1.0)

    """
    return MathOp("sin", (x,))


def sinh(x: OperableType) -> MathOp:
    """
    Hyperbolic sine of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import sinh
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = sinh(0)
    >>> r.toValue()
    np.float64(0.0)

    """
    return MathOp("sinh", (x,))


def tan(x: OperableType) -> MathOp:
    """
    Tangent of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import tan
    >>> import numpy as np
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = tan(np.pi/4)
    >>> round(r.toValue(), 2)
    np.float64(1.0)

    """
    return MathOp("tan", (x,))


def tanh(x: OperableType) -> MathOp:
    """
    Hyperbolic tangent of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import tanh
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = tanh(0)
    >>> r.toValue()
    np.float64(0.0)

    """
    return MathOp("tanh", (x,))


def acos(x: OperableType) -> MathOp:
    """
    Inverse cosine of ``x``.

    Returns
    -------
    Expresion | float

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import acos
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = acos(1)
    >>> r.toValue()
    np.float64(0.0)

    """
    return MathOp("arccos", (x,))


def asin(x: OperableType) -> MathOp:
    """
    Inver sinus of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import asin
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = asin(0)
    >>> r.toValue()
    np.float64(0.0)

    """
    return MathOp("arcsin", (x,))


def atan(x: OperableType) -> MathOp:
    """
    Inverse tangent of ``x``.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import atan
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = atan(0)
    >>> r.toValue()
    np.float64(0.0)

    """
    return MathOp("arctan", (x,))


def atan2(y: OperableType, x: OperableType) -> MathOp:
    """
    Four-quadrant arctan function yielding ``arctan(y/x)``, which is the angle the vector ``(x,y)`` makes with ``(1,0)`` in radians.

    Returns
    -------
    MathOp

    Examples
    --------
    >>> import math
    >>> from gamspy import Container, Parameter
    >>> from gamspy.math import atan2
    >>> m = Container()
    >>> r = Parameter(m, "r")
    >>> r[...] = atan2(1,1)
    >>> math.isclose(r.toValue(), 0.7853981633974483)
    True

    """
    return MathOp("arctan2", (y, x))
