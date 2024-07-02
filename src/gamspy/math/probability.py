from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
from gamspy.math.misc import MathOp

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.symbol import Symbol


def binomial(n: int | float | Symbol, k: int | float | Symbol) -> Expression:
    """
    (Generalized) Binomial coefficient for ``n > -1`` and ``-1 < k < n + 1``

    Parameters
    ----------
    n : int | float | Symbol
    k : int | float | Symbol

    Returns
    -------
    Expression

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
    return expression.Expression(None, MathOp("binomial", (n, k)), None)


def centropy(
    x: int | float | Symbol,
    y: int | float | Symbol,
    z: float = 1e-20,
) -> Expression:
    """
    Cross-entropy: ``x.ln((x + z) / (y + z))`` for ``x, y > 0`` and ``z >= 0``

    Parameters
    ----------
    x : float | Symbol
    y : float | Symbol
    z : float, optional

    Returns
    -------
    Expression

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

    return expression.Expression(None, MathOp("centropy", (x, y, z)), None)


def uniform(
    lower_bound: float | Expression,
    upper_bound: float | Expression,
) -> Expression:
    """
    Generates a random number from the uniform distribution between
    ``lower_bound`` and ``higher_bound``

    Parameters
    ----------
    lower_bound : float
    upper_bound : float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import uniform
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> x = Parameter(m, "x", domain=i, records=[("i1", 30), ("i2", 8), ("i3", 45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = uniform(x[i], 50)
    >>> b.toList()
    [('i1', 33.43494264), ('i2', 43.417201736), ('i3', 47.75187678)]

    """
    return expression.Expression(
        None, MathOp("uniform", (lower_bound, upper_bound)), None
    )


def uniformInt(
    lower_bound: int | float, upper_bound: int | float
) -> Expression:
    """
    Generates an integer random number from the discrete uniform distribution
    whose outcomes are the integers between ``lower_bound`` and ``higher_bound``

    Parameters
    ----------
    lower_bound : int | float
    upper_bound : int | float

    Returns
    -------
    Expression

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
    return expression.Expression(
        None,
        MathOp("uniformInt", (lower_bound, upper_bound)),
        None,
    )


def normal(mean: int | float, dev: int | float) -> Expression:
    """
    Generate a random number from the normal distribution with mean ``mean``
    and standard deviation ``dev``

    Parameters
    ----------
    mean : int | float
    dev : int | float

    Returns
    -------
    Expression

    Examples
    --------
    >>> from gamspy import Container, Set, Parameter
    >>> from gamspy.math import normal
    >>> m = Container()
    >>> i = Set(m, name="i", records=["i1", "i2", "i3"])
    >>> x = Parameter(m, "x", domain=i, records=[("i1", 30), ("i2", 8), ("i3", 45)])
    >>> b = Parameter(m, "b", domain=i)
    >>> b[i] = normal(x[i], 5)
    >>> b.toList()
    [('i1', 28.433285357057226), ('i2', 9.6383740411321), ('i3', 47.3177939118135)]

    """
    return expression.Expression(None, MathOp("normal", (mean, dev)), None)
