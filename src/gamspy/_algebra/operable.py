from __future__ import annotations

import math
import typing

import gamspy._algebra.expression as expression
import gamspy.math as gamspy_math

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

        sum_dim = self._validate_matrix_mult_dims(other)
        # TODO discuss let a be a parameter over domain [i, j]
        # does that ever make sense to write a itself as gamsRepr
        # as it needs to be domain controlled
        return operation.Sum([sum_dim], self * other)

    def _validate_matrix_mult_dims(self, other):
        """Validates the dimensions for the matrix multiplication"""
        left_len = len(self.domain)
        right_len = len(other.domain)
        assert left_len > 0, "Matrix multiplication requires at least 1 domain"
        assert (
            right_len > 0
        ), "Matrix multiplication requires at least 1 domain"

        lr = (left_len, right_len)
        if lr == (1, 1):
            # Dot product
            assert (
                self.domain[0] == other.domain[0]
            ), "Dot product requires same domain"
            return self.domain[0]
        elif lr == (2, 2):
            # Matrix multiplication
            assert (
                self.domain[1] == other.domain[0]
            ), "Matrix multiplication dimensions do not match"
            return self.domain[1]
        elif lr == (1, 2):
            # Vector matrix, vector 1-prepended
            assert (
                self.domain[0] == other.domain[0]
            ), "Matrix multiplication dimensions do not match"
            return self.domain[0]
        elif lr == (2, 1):
            # Matrix vector, ordinary
            assert (
                self.domain[1] == other.domain[0]
            ), "Matrix multiplication dimensions do not match"
            return self.domain[1]
        elif left_len == 1 and right_len > 2:
            # Vector batched-matrix, vector 1-prepended
            assert (
                self.domain[0] == other.domain[-2]
            ), "Matrix multiplication dimensions do not match"
            return self.domain[0]
        elif left_len > 2 and right_len == 1:
            # batched-matrix vector, ordinary
            assert (
                self.domain[-1] == other.domain[0]
            ), "Matrix multiplication dimensions do not match"
            return self.domain[-1]
        elif left_len > 2 and right_len > 2:
            # batched-matrix batched-matrix
            assert (
                self.domain[-1] == other.domain[-2]
            ), "Matrix multiplication dimensions do not match"
            return self.domain[-1]
        else:
            raise NotImplementedError(
                f"Matrix multiplication for left dim: {left_len},"
                " right dim: {right_len} not implemented"
            )
