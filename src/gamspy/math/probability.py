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


def binomial(
    n: Union[int, float, Symbol], k: Union[int, float, Symbol]
) -> Expression:
    """
    (Generalized) Binomial coefficient for n > -1, -1 < k < n + 1

    Parameters
    ----------
    n : int | float | Symbol
    k : int | float | Symbol

    Returns
    -------
    Expression
    """
    if isinstance(n, (int, float)) and isinstance(k, (int, float)):
        return expression.Expression(None, f"binomial({n},{k})", None)

    n_string = _stringify(n)
    k_string = _stringify(k)

    return expression.Expression(
        None, f"binomial({n_string},{k_string})", None
    )


def centropy(
    x: Union[int, float, Symbol],
    y: Union[int, float, Symbol],
    z: float = 1e-20,
) -> Expression:
    """
    Cross-entropy. x . ln((x + z) / (y + z)

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
    """
    if z < 0:
        raise ValueError("z must be greater than or equal to 0")

    x_str = _stringify(x)
    y_str = _stringify(y)

    return expression.Expression(None, f"centropy({x_str},{y_str},{z})", None)


def uniform(
    lower_bound: Union[float, Expression],
    upper_bound: Union[float, Expression],
) -> Expression:
    """
    Generates a random number from the uniform distribution between
    lower_bound and higher_bound

    Parameters
    ----------
    lower_bound : float
    upper_bound : float

    Returns
    -------
    Expression
    """
    lower_str = _stringify(lower_bound)
    upper_str = _stringify(upper_bound)
    return expression.Expression(
        None, f"uniform({lower_str},{upper_str})", None
    )


def uniformInt(
    lower_bound: Union[int, float], upper_bound: Union[int, float]
) -> Expression:
    """
    Generates an integer random number from the discrete uniform distribution
    whose outcomes are the integers between lower_bound and higher_bound.

    Parameters
    ----------
    lower_bound : int | float
    upper_bound : int | float
    Returns
    -------
    Expression
    """
    return expression.Expression(
        None,
        f"uniformInt({lower_bound},{upper_bound})",
        None,
    )


def normal(mean: Union[int, float], dev: Union[int, float]) -> Expression:
    """
    Generate a random number from the normal distribution with mean `mean`
    and `standard deviation` dev.

    Parameters
    ----------
    mean : int | float
    dev : int | float

    Returns
    -------
    Expression
    """
    return expression.Expression(None, f"normal({mean},{dev})", None)
