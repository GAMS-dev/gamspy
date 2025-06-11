from __future__ import annotations

import math
import typing

import gamspy._algebra.expression as expression
import gamspy.math as gamspy_math
from gamspy.exceptions import ValidationError

if typing.TYPE_CHECKING:
    from gamspy._types import OperableType


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
    def __add__(self, other: OperableType):
        from gamspy._symbols import Alias, Set
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (Alias, Set)) and not isinstance(
            other, (Alias, Set, ImplicitSet)
        ):
            return self.lead(other)

        if isinstance(
            self, (ImplicitSet, expression.SetExpression)
        ) or isinstance(other, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(self, "+", other)

        return expression.Expression(self, "+", other)

    def __radd__(self, other: OperableType):
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(
            self, (ImplicitSet, expression.SetExpression)
        ) or isinstance(other, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(other, "+", self)

        return expression.Expression(other, "+", self)

    def __sub__(self, other: OperableType):
        from gamspy._symbols import Alias, Set
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (Alias, Set)) and not isinstance(
            other, (Alias, Set, ImplicitSet)
        ):
            return self.lag(other)

        if isinstance(
            self, (ImplicitSet, expression.SetExpression)
        ) or isinstance(other, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(self, "-", other)

        return expression.Expression(self, "-", other)

    def __rsub__(self, other: OperableType):
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(
            self, (ImplicitSet, expression.SetExpression)
        ) or isinstance(other, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(other, "-", self)

        return expression.Expression(other, "-", self)

    def __neg__(self):
        return expression.Expression(None, "-", self)

    def __truediv__(self, other: OperableType):
        return expression.Expression(self, "/", other)

    def __rtruediv__(self, other: OperableType):
        return expression.Expression(other, "/", self)

    def __mul__(self, other: OperableType):
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(
            self, (ImplicitSet, expression.SetExpression)
        ) or isinstance(other, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(self, "*", other)

        return expression.Expression(self, "*", other)

    def __rmul__(self, other: OperableType):
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(
            self, (ImplicitSet, expression.SetExpression)
        ) or isinstance(other, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(other, "*", self)

        return expression.Expression(other, "*", self)

    @typing.no_type_check
    def __pow__(self, other: OperableType):
        if (
            isinstance(other, int)
            and other == 2
            and isinstance(self, expression.Expression)
            and isinstance(self.left, gamspy_math.misc.MathOp)
            and self.left.op_name == "sqrt"
            and self.left.safe_cancel
        ):
            return self.left.elements[0]

        if isinstance(other, int):
            return gamspy_math.power(self, other)
        elif isinstance(other, float):
            if other == 0.5:
                return gamspy_math.sqrt(self)
            elif math.isclose(other, round(other), rel_tol=1e-4):
                return gamspy_math.power(self, other)

        return gamspy_math.rpower(self, other)

    def __rpow__(self, other: int | float):
        # e.g. 2 ** a[i] -> where 2 is other and a[i] is self.
        return gamspy_math.rpower(other, self)

    def __mod__(self, other: OperableType):
        return gamspy_math.mod(self, other)

    # and, or, xor
    def __and__(self, other: OperableType):
        return expression.Expression(self, "and", other)

    def __rand__(self, other: OperableType):
        return expression.Expression(other, "and", self)

    def __or__(self, other: OperableType):
        return expression.Expression(self, "or", other)

    def __ror__(self, other: OperableType):
        return expression.Expression(other, "or", self)

    def __xor__(self, other: OperableType):
        return expression.Expression(self, "xor", other)

    def __rxor__(self, other: OperableType):
        return expression.Expression(other, "xor", self)

    # <, <=, >, >=, ==, !=
    def __lt__(self, other: OperableType):
        return expression.Expression(self, "<", other)

    def __le__(self, other: OperableType):
        return expression.Expression(self, "=l=", other)

    def __gt__(self, other: OperableType):
        return expression.Expression(self, ">", other)

    def __ge__(self, other: OperableType):
        return expression.Expression(self, "=g=", other)

    # ~ -> not
    def __invert__(self):
        from gamspy._symbols.implicits import ImplicitSet

        if isinstance(self, (ImplicitSet, expression.SetExpression)):
            return expression.SetExpression(None, "not", self)
        return expression.Expression(None, "not", self)

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

    def gamsRepr(self):
        """Representation of the symbol in GAMS"""
