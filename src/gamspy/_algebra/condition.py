from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._symbols as syms
import gamspy._symbols.implicits as implicits
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._symbols.implicits import (
        ImplicitParameter,
        ImplicitSet,
    )
    from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol


class Condition:
    """
    Condition class allows symbols to be conditioned.

    Parameters
    ----------
    symbol: ImplicitSymbol | Expression
        Reference to the symbol to be conditioned.

    >>> muf[i, j] = (2.48 + 0.0084 * rd[i, j]).where[rd[i, j]]
    >>> minw[t].where[tm[t]] = Sum(w.where[td[w, t]], x[w, t]) >= tm[t]
    """

    def __init__(self, symbol: ImplicitSymbol | Expression):
        self._symbol = symbol

    def __getitem__(
        self, condition: Expression | ImplicitParameter | ImplicitSet
    ) -> Expression:
        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        statement = expression.Expression(self._symbol, "$", condition)
        return statement

    def __setitem__(self, condition, rhs):
        # symbol.where[condition] = rhs
        eq_types = (syms.Equation, implicits.ImplicitEquation)
        if isinstance(rhs, bool):
            rhs = "yes" if rhs is True else "no"

        op_type = ".." if isinstance(self._symbol, eq_types) else "="

        if isinstance(condition, expression.Expression):
            condition._fix_equalities()

        lhs = expression.Expression(self._symbol, "$", condition)
        statement = expression.Expression(lhs, op_type, rhs)

        self._symbol.container._add_statement(statement)

        if isinstance(self._symbol, ImplicitSymbol):
            self._symbol.container[self._symbol.parent.name]._is_dirty = True
            self._symbol.parent._assignment = statement

        if isinstance(self._symbol, implicits.ImplicitEquation):
            self._symbol.parent._definition = statement

        self._symbol.container._run()
