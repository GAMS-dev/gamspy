from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
import gamspy.utils as utils
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy import Alias, Parameter, Set, Variable
    from gamspy._algebra.domain import Domain
    from gamspy._algebra.expression import Expression
    from gamspy._algebra.number import Number
    from gamspy._algebra.operation import Card, Operation, Ord
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
    )


class Condition(operable.Operable):
    """
    Condition class allows conditioning on GAMSPy constructs such as Symbols and Expressions.

    Parameters
    ----------
    symbol: ImplicitSymbol | Expression
        Reference to the symbol to be conditioned.

    >>> muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    >>> minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    """

    def __init__(
        self,
        conditioning_on: ImplicitSymbol
        | Set
        | Alias
        | Parameter
        | Variable
        | Expression
        | Operation
        | Domain
        | Number
        | Card
        | Ord,
        condition: Expression | ImplicitParameter | ImplicitSet | None = None,
    ):
        self.conditioning_on = conditioning_on
        self.condition = condition

    def __getitem__(
        self, condition: Expression | ImplicitParameter | ImplicitSet
    ) -> Condition:
        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        return Condition(self.conditioning_on, condition)

    def __setitem__(self, condition, rhs):
        # conditioning_on.where[condition] = rhs
        eq_types = (syms.Equation, implicits.ImplicitEquation)
        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"

        op_type = ".." if isinstance(self.conditioning_on, eq_types) else "="

        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        lhs = Condition(self.conditioning_on, condition)
        statement = expression.Expression(lhs, op_type, rhs)

        if isinstance(self.conditioning_on, ImplicitSymbol):
            statement._validate_definition(
                utils._unpack(self.conditioning_on.domain)
            )

        self.conditioning_on.container._add_statement(statement)

        if isinstance(self.conditioning_on, ImplicitSymbol):
            self.conditioning_on.parent._assignment = statement

        if isinstance(self.conditioning_on, implicits.ImplicitEquation):
            self.conditioning_on.parent._definition = statement

        self.conditioning_on.container._synch_with_gams()

    def gamsRepr(self) -> str:
        assert self.condition is not None
        return f"({self.conditioning_on.gamsRepr()} $ {self.condition.gamsRepr()})"

    def getDeclaration(self) -> str:
        return self.gamsRepr()

    def latexRepr(self) -> str:
        """
        Representation of this condition in Latex.

        Returns
        -------
        str
        """
        assert self.condition is not None

        return f"{self.conditioning_on.latexRepr()} ~ | ~ {self.condition.latexRepr()}"
