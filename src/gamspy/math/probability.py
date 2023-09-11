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


def binomial(
    n: Union[int, float, "Symbol"], k: Union[int, float, "Symbol"]
) -> "Expression":
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
        return expression.Expression("binomial(", f"{n},{k}", ")")

    n_string = str(n) if isinstance(n, (int, float)) else n.gamsRepr()
    k_string = str(k) if isinstance(k, (int, float)) else k.gamsRepr()

    return expression.Expression("binomial(", f"{n_string},{k_string}", ")")


def centropy(
    x: "Symbol",
    y: "Symbol",
    z: float = 1e-20,
) -> "Expression":
    """
    Cross-entropy. x . ln((x + z) / (y + z)

    Parameters
    ----------
    x : float | Operable
    y : float | Operable
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

    return expression.Expression(
        "centropy(", ",".join([x.gamsRepr(), y.gamsRepr(), str(z)]), ")"
    )


def uniform(lower_bound: float, upper_bound: float) -> "Expression":
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
    return expression.Expression(
        "uniform(", f"{lower_bound},{upper_bound}", ")"
    )


def uniformInt(
    lower_bound: Union[int, float], upper_bound: Union[int, float]
) -> "Expression":
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
        "uniformInt(", f"{lower_bound},{upper_bound}", ")"
    )


def normal(mean: Union[int, float], dev: Union[int, float]) -> "Expression":
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
    return expression.Expression("normal(", f"{mean},{dev}", ")")
