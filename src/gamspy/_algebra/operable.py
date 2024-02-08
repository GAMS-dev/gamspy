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

        left_domain, right_domain, sum_domain = (
            self._validate_matrix_mult_dims(other)
        )
        return operation.Sum(
            [sum_domain], self[left_domain] * other[right_domain]
        )

    def _validate_matrix_mult_dims(self, other):
        """Validates the dimensions for the matrix multiplication"""
        from gamspy.math import next_alias

        left_len = len(self.domain)
        right_len = len(other.domain)

        dim_no_match_err = "Matrix multiplication dimensions do not match"

        if left_len == 0:
            raise ValidationError(
                "Matrix multiplication requires at least 1 domain, left side"
                " is a scalar"
            )

        if right_len == 0:
            raise ValidationError(
                "Matrix multiplication requires at least 1 domain, right side"
                " is a scalar"
            )

        lr = (left_len, right_len)

        unique_check_list = []
        matrix_dim_left = self.domain[-2:]
        if len(matrix_dim_left) == 2:
            unique_check_list.append(self.domain[-2])

        matrix_dim_right = other.domain[-2:]
        if len(matrix_dim_right) == 2:
            unique_check_list.append(other.domain[-1])

        if lr == (1, 1):
            # Dot product
            if self.domain[0] != other.domain[0]:
                raise ValidationError("Dot product requires same domain")

            return ..., ..., self.domain[0]
        elif lr == (2, 2):
            # Matrix multiplication
            if self.domain[1] != other.domain[0]:
                raise ValidationError(dim_no_match_err)

            sum_domain = self.domain[1]
            while sum_domain in unique_check_list:
                sum_domain = next_alias(sum_domain)

            return (
                [self.domain[0], sum_domain],
                [sum_domain, other.domain[1]],
                sum_domain,
            )
        elif lr == (1, 2):
            # Vector matrix, vector 1-prepended
            if self.domain[0] != other.domain[0]:
                raise ValidationError(dim_no_match_err)

            sum_domain = self.domain[0]
            if other.domain[0] == other.domain[1]:
                sum_domain = next_alias(sum_domain)

            return [sum_domain], [sum_domain, other.domain[1]], sum_domain
        elif lr == (2, 1):
            # Matrix vector, ordinary
            if self.domain[1] != other.domain[0]:
                raise ValidationError(dim_no_match_err)

            sum_domain = self.domain[1]
            if self.domain[0] == self.domain[1]:
                sum_domain = next_alias(sum_domain)

            return [self.domain[0], sum_domain], [sum_domain], sum_domain
        elif left_len == 1 and right_len > 2:
            # Vector batched-matrix, vector 1-prepended
            if self.domain[0] != other.domain[-2]:
                raise ValidationError(dim_no_match_err)

            sum_domain = self.domain[0]
            if other.domain[-2] == other.domain[-1]:
                sum_domain = next_alias(sum_domain)

            return (
                [sum_domain],
                [*other.domain[:-2], sum_domain, other.domain[-1]],
                sum_domain,
            )
        elif left_len > 2 and right_len == 1:
            # batched-matrix vector, ordinary
            if self.domain[-1] != other.domain[0]:
                raise ValidationError(dim_no_match_err)

            sum_domain = self.domain[-1]
            if self.domain[-1] == self.domain[-2]:
                sum_domain = next_alias(sum_domain)

            return (
                [*self.domain[:-2], sum_domain, self.domain[-1]],
                [sum_domain],
                sum_domain,
            )
        elif left_len >= 2 and right_len >= 2:
            # batched-matrix batched-matrix
            if self.domain[-1] != other.domain[-2]:
                raise ValidationError(dim_no_match_err)

            batch_dim_1 = self.domain[:-2]
            batch_dim_2 = other.domain[:-2]

            if len(batch_dim_1) != len(batch_dim_2):
                raise ValidationError("Batch dimensions do not match")

            if any([x != y for x, y in zip(batch_dim_1, batch_dim_2)]):
                raise ValidationError("Batch dimensions do not match")

            sum_domain = self.domain[-1]
            while sum_domain in unique_check_list:
                sum_domain = next_alias(sum_domain)

            return (
                [*self.domain[:-1], sum_domain],
                [*other.domain[:-2], sum_domain, other.domain[-1]],
                sum_domain,
            )
        else:
            raise ValidationError(
                f"Matrix multiplication for left dim: {left_len},"
                f" right dim: {right_len} not implemented"
            )
