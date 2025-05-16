from __future__ import annotations

from typing import TYPE_CHECKING

from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._types import OperableType


def binomial(n: OperableType, k: OperableType) -> MathOp:
    """
    (Generalized) Binomial coefficient for ``n > -1`` and ``-1 < k < n + 1``

    Parameters
    ----------
    n : OperableType
    k : OperableType

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import binomial
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> p = Parameter(m, "p", domain=i, records=[("i1", 0.3), ("i2", 0.8), ("i3", 0.45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = binomial(75, p[i])

    """
    return MathOp("binomial", (n, k))


def centropy(
    x: OperableType,
    y: OperableType,
    z: float = 1e-20,
) -> MathOp:
    """
    Cross-entropy: ``x.ln((x + z) / (y + z))`` for ``x, y > 0`` and ``z >= 0``

    Parameters
    ----------
    x : OperableType
    y : OperableType
    z : float, optional

    Returns
    -------
    MathOp

    Raises
    ------
    ValueError
        if z is smaller than 0

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import centropy
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> x = Parameter(m, "x", domain=i, records=[("i1", 0.3), ("i2", 8), ("i3", 45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = centropy(2.8, x[i])
    >>> b.toList()
    [('i1', 6.254058220219863), ('i2', -2.939501948596297), ('i3', -7.775720603249651)]

    """
    if not isinstance(z, (int, float)):
        raise TypeError("z must be a number")

    if z < 0:
        raise ValueError("z must be greater than or equal to 0")

    return MathOp("centropy", (x, y, z))


def uniform(
    lower_bound: float | Expression,
    upper_bound: float | Expression,
) -> MathOp:
    """
    Generates a random number from the uniform distribution between
    ``lower_bound`` and ``higher_bound``

    Parameters
    ----------
    lower_bound : float
    upper_bound : float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import uniform
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> x = Parameter(m, "x", domain=i, records=[("i1", 30), ("i2", 8), ("i3", 45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = uniform(x[i], 50)

    """
    return MathOp("uniform", (lower_bound, upper_bound))


def uniformInt(lower_bound: int | float, upper_bound: int | float) -> MathOp:
    """
    Generates an integer random number from the discrete uniform distribution
    whose outcomes are the integers between ``lower_bound`` and ``higher_bound``

    Parameters
    ----------
    lower_bound : int | float
    upper_bound : int | float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import uniformInt
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> x = Parameter(m, "x", domain=i, records=[("i1", 30), ("i2", 8), ("i3", 45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = uniformInt(x[i], 50)
    >>> b.toList()
    [('i1', 33.0), ('i2', 44.0), ('i3', 48.0)]

    """
    return MathOp("uniformInt", (lower_bound, upper_bound))


def normal(mean: int | float, dev: int | float) -> MathOp:
    """
    Generate a random number from the normal distribution with mean ``mean``
    and standard deviation ``dev``

    Parameters
    ----------
    mean : int | float
    dev : int | float

    Returns
    -------
    MathOp

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import normal
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> x = Parameter(m, "x", domain=i, records=[("i1", 30), ("i2", 8), ("i3", 45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = normal(x[i], 5)

    """
    return MathOp("normal", (mean, dev))
