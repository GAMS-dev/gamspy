from __future__ import annotations

import math
import typing

import gamspy._algebra.expression as expression
import gamspy.math as gamspy_math

from gamspy.exceptions import ValidationError

if typing.TYPE_CHECKING:
    from gamspy import Alias, Equation, Parameter, Set, Variable
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
        ImplicitVariable,
    )

    OperableType = typing.Union[
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

    @typing.no_type_check
    def __pow__(self: OperableType, other: OperableType):
        if (
            other == 2
            and isinstance(self, expression.Expression)
            and isinstance(self.data, gamspy_math.misc.MathOp)
            and self.data.op_name == "sqrt"
            and self.data.safe_cancel
        ):
            return self.data.elements[0]

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

    # a @ b
    def __matmul__(self, other):
        import gamspy._algebra.operation as operation
        from gamspy.math.matrix import _validate_matrix_mult_dims

        left_domain, right_domain, sum_domain = _validate_matrix_mult_dims(
            self, other
        )
        return operation.Sum(
            [sum_domain], self[left_domain] * other[right_domain]
        )
