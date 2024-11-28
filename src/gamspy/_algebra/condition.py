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
        condition: Expression
        | ImplicitParameter
        | ImplicitSet
        | int
        | None = None,
    ):
        self.conditioning_on = conditioning_on
        self.condition = condition
        self._where = None

    @property
    def where(self):
        return Condition(self)

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

        self.conditioning_on.container._synch_with_gams(gams_to_gamspy=True)

    def __neg__(self) -> Expression:
        return expression.Expression(None, "-", self)

    def __repr__(self) -> str:
        return f"Condition(conditioning_on={self.conditioning_on}, condition={self.condition})"

    def gamsRepr(self) -> str:
        condition_str = (
            self.condition.gamsRepr()  # type: ignore
            if hasattr(self.condition, "gamsRepr")
            else str(self.condition)
        )
        return f"({self.conditioning_on.gamsRepr()} $ {condition_str})"  # type: ignore

    def getDeclaration(self) -> str:
        return self.gamsRepr()

    def latexRepr(self) -> str:
        """
        Representation of this condition in Latex.

        Returns
        -------
        str
        """
        condition_str = (
            self.condition.latexRepr()  # type: ignore
            if hasattr(self.condition, "latexRepr")
            else str(self.condition)
        )
        return f"{self.conditioning_on.latexRepr()} ~ | ~ {condition_str}"  # type: ignore
