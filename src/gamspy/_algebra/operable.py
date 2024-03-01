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
import math

from typing import TYPE_CHECKING, Union

import gamspy.math as gamspy_math
import gamspy._algebra.expression as expression

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Operation
    from gamspy._algebra.operation import Ord, Card
    from gamspy import Alias, Equation, Parameter, Set, Variable
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )

    OperableType = Union[
        Alias,
        Equation,
        Parameter,
        Set,
        Variable,
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
        Expression,
        Number,
        Operation,
        Ord,
        Card,
    ]


class Operable:
    """
    A mixin class that overloads the magic operations of a class
    to be used in Expressions
    """

    # +, -, /, *, **
    def __add__(self: OperableType, other: OperableType):
        return expression.Expression(self, "+", other)

    def __radd__(self: OperableType, other: OperableType):
        return expression.Expression(other, "+", self)

    def __sub__(self: OperableType, other: OperableType):
        return expression.Expression(self, "-", other)

    def __rsub__(self: OperableType, other: OperableType):
        return expression.Expression(other, "-", self)

    def __truediv__(self: OperableType, other: OperableType):
        return expression.Expression(self, "/", other)

    def __rtruediv__(self: OperableType, other: OperableType):
        return expression.Expression(other, "/", self)

    def __mul__(self: OperableType, other: OperableType):
        return expression.Expression(self, "*", other)

    def __rmul__(self: OperableType, other: OperableType):
        return expression.Expression(other, "*", self)

    def __pow__(self: OperableType, other: OperableType):
        if isinstance(other, int):
            return gamspy_math.power(self, other)
        elif isinstance(other, float):
            if other == 0.5:
                return gamspy_math.sqrt(self)
            elif math.isclose(other, round(other), rel_tol=1e-4):
                return gamspy_math.power(self, other)

        return gamspy_math.rpower(self, other)

    # not, and, or, xor
    def __and__(self: OperableType, other: OperableType):
        return expression.Expression(self, "and", other)

    def __rand__(self: OperableType, other: OperableType):
        return expression.Expression(other, "and", self)

    def __or__(self: OperableType, other: OperableType):
        return expression.Expression(self, "or", other)

    def __ror__(self: OperableType, other: OperableType):
        return expression.Expression(other, "or", self)

    def __xor__(self: OperableType, other: OperableType):
        return expression.Expression(self, "xor", other)

    def __rxor__(self: OperableType, other: OperableType):
        return expression.Expression(other, "xor", self)

    # <, <=, >, >=, ==, !=
    def __lt__(self: OperableType, other: OperableType):
        return expression.Expression(self, "<", other)

    def __le__(self: OperableType, other: OperableType):
        return expression.Expression(self, "=l=", other)

    def __gt__(self: OperableType, other: OperableType):
        return expression.Expression(self, ">", other)

    def __ge__(self: OperableType, other: OperableType):
        return expression.Expression(self, "=g=", other)

    # ~ -> not
    def __invert__(self: OperableType):
        return expression.Expression("", "not", self)
