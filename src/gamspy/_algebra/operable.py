from __future__ import annotations

import typing
from abc import abstractmethod

import numpy as np

import gamspy._algebra.expression as expression
import gamspy.math as gamspy_math
from gamspy._config import get_option
from gamspy.exceptions import ValidationError

if typing.TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.operation import Sum
    from gamspy._symbols import Parameter, Variable
    from gamspy._symbols.implicits import ImplicitParameter, ImplicitVariable
    from gamspy._types import OperableType
    from gamspy.math.misc import MathOp


class Operable:
    """
    A mixin class that overloads the magic operations of a class
    to be used in Expressions
    """

    def __iter__(self):
        raise ValidationError(
            "GAMSPy symbols are not iterable. If you want to iterate on records, iterate over <symbol>.records."
        )

    # +, -, /, *, **, %
    def __add__(self, other: OperableType) -> Expression:
        from gamspy._symbols import Alias, Set
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (Alias, Set)) and not isinstance(
            other, (Alias, Set, ImplicitSet, expression.ShiftExpression)
        ):
            return self.lead(other)

        if isinstance(self, (ImplicitSet, expression.SetExpression)) or isinstance(
            other, (ImplicitSet, expression.SetExpression)
        ):
            return expression.SetExpression(self, "+", other)

        return expression.Expression(self, "+", other)

    def __radd__(self, other: OperableType) -> Expression:
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (ImplicitSet, expression.SetExpression)) or isinstance(
            other, (ImplicitSet, expression.SetExpression)
        ):
            return expression.SetExpression(other, "+", self)

        return expression.Expression(other, "+", self)

    def __sub__(self, other: OperableType) -> Expression:
        from gamspy._symbols import Alias, Set
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (Alias, Set)) and not isinstance(
            other, (Alias, Set, ImplicitSet, expression.ShiftExpression)
        ):
            return self.lag(other)

        if isinstance(self, (ImplicitSet, expression.SetExpression)) or isinstance(
            other, (ImplicitSet, expression.SetExpression)
        ):
            return expression.SetExpression(self, "-", other)

        return expression.Expression(self, "-", other)

    def __rsub__(self, other: OperableType) -> Expression:
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (ImplicitSet, expression.SetExpression)) or isinstance(
            other, (ImplicitSet, expression.SetExpression)
        ):
            return expression.SetExpression(other, "-", self)

        return expression.Expression(other, "-", self)

    def __neg__(self) -> Expression:
        return expression.Expression(None, "u-", self)

    def __truediv__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "/", other)

    def __rtruediv__(self, other: OperableType) -> Expression:
        return expression.Expression(other, "/", self)

    def __mul__(self, other: OperableType) -> Expression:
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (ImplicitSet, expression.SetExpression)) or isinstance(
            other, (ImplicitSet, expression.SetExpression)
        ):
            return expression.SetExpression(self, "*", other)

        return expression.Expression(self, "*", other)

    def __rmul__(self, other: OperableType) -> Expression:
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (ImplicitSet, expression.SetExpression)) or isinstance(
            other, (ImplicitSet, expression.SetExpression)
        ):
            return expression.SetExpression(other, "*", self)

        return expression.Expression(other, "*", self)

    @typing.no_type_check
    def __pow__(self, other: OperableType) -> Expression:
        # The operation x**y is equivalent to the function rPower(x,y) and is calculated
        # internally as e^(y x log(x)). This operation is not defined if x is negative.
        # If the possibility of negative values for x is to be admitted and the exponent
        # is known to be an integer, then the function power(x,n) may be used.
        # https://gams.com/latest/docs/UG_Parameters.html#UG_Parameters_Expressions
        if isinstance(other, (bool, np.integer)) or (
            isinstance(other, float) and other.is_integer()
        ):
            other = int(other)

        if (
            isinstance(other, int)
            and other == 2
            and isinstance(self, expression.Expression)
            and isinstance(self.left, gamspy_math.misc.MathOp)
            and self.left.op_name == "sqrt"
            and self.left.safe_cancel
        ):
            return self.left.elements[0]

        if get_option("STRICT_POWER_OPERATOR"):
            return gamspy_math.rpower(self, other)

        if isinstance(other, int):
            return gamspy_math.power(self, other)
        elif isinstance(other, float) and other == 0.5:
            return gamspy_math.sqrt(self)

        return gamspy_math.rpower(self, other)

    def __rpow__(self, other: int | float) -> Expression:
        # e.g. 2 ** a[i] -> where 2 is other and a[i] is self.
        return gamspy_math.rpower(other, self)

    def __mod__(self, other: OperableType) -> MathOp:
        return gamspy_math.mod(self, other)

    # and, or, xor
    def __and__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "and", other)

    def __rand__(self, other: OperableType) -> Expression:
        return expression.Expression(other, "and", self)

    def __or__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "or", other)

    def __ror__(self, other: OperableType) -> Expression:
        return expression.Expression(other, "or", self)

    def __xor__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "xor", other)

    def __rxor__(self, other: OperableType) -> Expression:
        return expression.Expression(other, "xor", self)

    # <, <=, >, >=, ==, !=
    def __lt__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "<", other)

    def __le__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "=l=", other)

    def __gt__(self, other: OperableType) -> Expression:
        return expression.Expression(self, ">", other)

    def __ge__(self, other: OperableType) -> Expression:
        return expression.Expression(self, "=g=", other)

    # ~ -> not
    def __invert__(self) -> Expression:
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(None, "not", self)
        return expression.Expression(None, "not", self)

    # a @ b
    def __matmul__(
        self: Parameter | Variable | ImplicitParameter | ImplicitVariable,
        other: Parameter | Variable | ImplicitParameter | ImplicitVariable,
    ) -> Sum:
        import gamspy._algebra.operation as operation
        from gamspy.math.matrix import _validate_matrix_mult_dims

        left_domain, right_domain, sum_domain = _validate_matrix_mult_dims(self, other)
        return operation.Sum([sum_domain], self[left_domain] * other[right_domain])

    def __str__(self):
        return self.gamsRepr()

    @abstractmethod
    def gamsRepr(self) -> str:
        """Representation of the symbol in GAMS"""
